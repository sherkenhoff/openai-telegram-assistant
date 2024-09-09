#!/usr/bin/env python3

# Import the standard libraries
import base64
import json
import queue
import sqlite3
import threading
import time
import requests
import urllib
import os
import datetime
import sys

# Import the OpenAI library
from openai import OpenAI

# Variables related to the Telegram bot, passed as environment variables from the caller
TOKEN = os.environ['TELEGRAM_TOKEN']
URL = "https://api.telegram.org/bot{}/".format(TOKEN)
TELEGRAM_SENDPHOTO = "https://api.telegram.org/bot{}/sendPhoto".format(TOKEN)
TELEGRAM_SENDVOICE = "https://api.telegram.org/bot{}/sendVoice".format(TOKEN)
TELEGRAM_GETFILE = "https://api.telegram.org/bot{}/getFile".format(TOKEN)
TELEGRAM_FILE = "https://api.telegram.org/file/bot{}/".format(TOKEN)
TELEGRAM_ADMIN_ID = os.environ['TELEGRAM_ADMIN_ID']

# Variables related to files
CHATBOT_DATA_PATH = os.environ['CHATBOT_DATA_PATH']
IMAGE_FILE_PATH = os.path.join(CHATBOT_DATA_PATH, 'images')
TRANSCRIPTION_FILE_PATH = os.path.join(CHATBOT_DATA_PATH, 'transcriptions')
DB_FILE = os.path.join(CHATBOT_DATA_PATH, 'chatbot.sqlite')

# Variables related to the regular maintenance tasks
MATINTENANCE_TASKS_INTERVAL = int(os.environ['CHATBOT_MATINTENANCE_INTERVAL_SECONDS'])

# variables related to the message history
MESSAGE_HISTORY_TIMEOUT_MINUTES = 3600 # 24h
IMAGES_TO_KEEP_PER_CHATID = 10

# lock to prevent concurrent write access to the database
db_write_lock = threading.Lock()

# initialize OpenAI client
openai_client = OpenAI()

# OpenAI chat variables
openai_default_chat_model = "gpt-3.5-turbo-0125"

# OpenAI DALL-E variables
openai_image_model = "dall-e-3" # the name of the image model to use, "dall-e-3", "dall-e-2.1", "dall-e-2"
openai_image_size = "1024x1024" # the size of the image to generate, "256x256", "512x512", "1024x1024"

# chat queues
active_chats = dict()

# helper function to print and flush a message
def print_flush(message):
    print(message)
    sys.stdout.flush()

# Define a custom exception for message list handling
class MessageClearedException(Exception):
    pass

# Function to generate an image using OpenAI's DALL-E model
# - prompt is the prompt to use for generating the image
# - quality is the quality of the image to generate, can be either standard or hd
# - chatid is the chatid of the user to send the image to
# - stores the image on the filesystem and in the database
# - returns a JSON object with the prompt, the URL of the image, and the duration of the function
def render_dalle_image(prompt, quality, chatid):
    """Generate an image using OpenAI's DALL-E model"""
    global db_conn, db_cursor
    print_flush('render_dalle_image({}, {})'.format(prompt, quality))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    t_in = datetime.datetime.now()
    try:
        response = openai_client.images.generate(
            model = openai_image_model,
            prompt = prompt,
            size = openai_image_size,
            quality = quality,
            response_format = "b64_json",
            n = 1
        )
    except Exception as e:
        print_flush("Error generating image: {}".format(str(e)))
        return json.dumps({"prompt": prompt, "error": str(e)})
    t_duration = datetime.datetime.now() - t_in
    # convert the base64 to a png and store it on the filesystem and in the database
    # - image_filename is timestamp.png
    # - image path is under IMAGE_FILE_PATH
    image_filename = "{}.png".format(timestamp)
    try:
        image_folder = os.path.join(IMAGE_FILE_PATH, str(chatid))
        image_path = os.path.join(image_folder, image_filename)
    except Exception as e:
        print_flush("Error creating image path: {}".format(str(e)))
        return json.dumps({"prompt": prompt, "error": str(e)})
    # make sure the directory exists
    try:
        os.makedirs(image_folder, exist_ok=True)
    except Exception as e:
        print_flush("Error creating directory for image: {}".format(str(e)))
        return json.dumps({"prompt": prompt, "error": str(e)})
    # check if the image exists, return an error if it does
    if os.path.exists(image_path):
        print_flush("Error: Image already exists: {}".format(image_path))
        return json.dumps({"prompt": prompt, "error": "Image already exists: {}".format(image_path)})
    # store the image on the filesystem
    try:
        with open(image_path, "wb") as img_file:
            img_file.write(base64.b64decode(response.data[0].b64_json))
    except Exception as e:
        print_flush("Error storing image (path: {}) on the filesystem: {}".format(image_path, str(e)))
        return json.dumps({"prompt": prompt, "error": str(e)})
    # store the image data in the database
    try:
        db_write_lock.acquire()
        db_cursor.execute('INSERT INTO images (chatid, image_filename, timestamp_created, prompt, revised_prompt) VALUES (?, ?, ?, ?, ?)', (chatid, image_filename, timestamp, prompt, response.data[0].revised_prompt))
        db_conn.commit()
        db_write_lock.release()
    except sqlite3.Error as e:
        print_flush("Error storing image data in the database: {}".format(str(e)))
        return json.dumps({"prompt": prompt, "error": str(e)})
    # send the image to the user
    try:
        with open(image_path, "rb") as img_file:
            status = requests.post(TELEGRAM_SENDPHOTO, files={'photo': img_file}, data={'chat_id': chatid})
    except Exception as e:
        print_flush("Error sending image to user: {}".format(str(e)))
        return json.dumps({"prompt": prompt, "error": str(e), "duration": t_duration.seconds})
    return json.dumps({"revised prompt": response.data[0].revised_prompt, "status": "Image generated and sent do user. HTTP response={}".format(status.status_code), "duration": t_duration.seconds})

# Function to generate text to speech using OpenAI's TTS model
def generate_text_to_speech(chatid, text, voice="onyx"):
    """Generate text to speech using OpenAI's TTS model"""
    print_flush('generate_text_to_speech(chatid={}, text={}, voice={})'.format(chatid, text, voice))
    # Open a file to write the TTS audio to using the current timestamp
    audio_file_path = "/tmp/openai-tts-{}.aac".format(datetime.datetime.now().timestamp())
    t_in = datetime.datetime.now()
    # request the TTS from OpenAI
    try:
        response = openai_client.audio.speech.create(
            model = "tts-1",
            response_format = "aac",
            voice = voice,
            input = text
        )
    except Exception as e:
        print_flush("Error generating TTS audio: {}".format(str(e)))
        return json.dumps({"status": "ERROR: {}".format(str(e)), "duration": 0})
    # save the TTS to the file opened earlier
    try:
        with open(audio_file_path, "wb") as audio_file:
            audio_file.write(response.content)
    except Exception as e:
        print_flush("Error saving TTS audio file: {}".format(str(e)))
        return json.dumps({"status": "ERROR: {}".format(str(e)), "duration": 0})
    
    # send the TTS to the user
    response = requests.post(TELEGRAM_SENDVOICE, files={'voice': open(audio_file_path, 'rb')}, data={'chat_id': chatid})
    
    # remove the TTS file
    os.remove(audio_file_path)
    t_duration = datetime.datetime.now() - t_in

    return json.dumps({"status": "Generated TTS successfully and sent to user.", "duration": t_duration.seconds})

# Function to set the GPT model to use for the chat
def gpt_model(chatid, model = None):
    """Set the GPT model to use for the chat. If called with empty parameter 'model', it will return the current model."""
    print_flush('gpt_model({})'.format(model))
    if model:
        active_chats[chatid]['model'] = model
        return json.dumps({"status": "Model set to {}.".format(model)})
    else:
        return json.dumps({"status": "Current model is {}.".format(active_chats[chatid]['model'])})

# Function to store an item in the items list
def add_thing_to_items_list(item, owner, quantity, chatid):
    """Stores an item in the items list. Needs the name of the item and the owner of the the item. Optionally provide the quantity of the item owned."""
    global db_conn, db_cursor
    print_flush('add_thing_to_items_list({}, {}, {}, {})'.format(item, owner, quantity, chatid))
    # lock the database for writing
    db_write_lock.acquire()
    try:
        # insert the item into the database
        db_cursor.execute('INSERT INTO items (chatid, item, owner, quantity) VALUES (?, ?, ?, ?)', (chatid, item, owner, quantity))
        db_conn.commit()
    except sqlite3.Error as e:
        print_flush('Error adding item to the items list: {}'.format(str(e)))
        return json.dumps({"status": "ERROR: {}".format(str(e))})
    finally:
        # release the lock
        db_write_lock.release()
    return json.dumps({"status": "Item added to the items list."})

# Function to show the list of items stored in the items list
def show_items_list(owner, chatid):
    """Shows the list of items stored in the items list. Optionally provide the owner of the items to show only the items owned by the owner."""
    global db_conn, db_cursor
    print_flush('show_items_list({}, {})'.format(owner, chatid))
    try:
        # select the items from the database
        if owner:
            db_cursor.execute('SELECT * FROM items WHERE chatid = ? AND owner = ?', (chatid, owner))
        else:
            db_cursor.execute('SELECT * FROM items WHERE chatid = ?', (chatid,))
        items = db_cursor.fetchall()
    except sqlite3.Error as e:
        print_flush('Error showing items list: {}'.format(str(e)))
        return json.dumps({"status": "ERROR: {}".format(str(e))})
    return json.dumps({"status": "Items list shown.", "items": items})

def get_url(url):
    try:
        response = requests.get(url, timeout=50)
    except:
        print_flush('get_url failed')
        print_flush('url: {}'.format(url))
        return None
    content = response.content.decode("utf8")
    return content

def get_updates(offset=None):
    url = URL + "getUpdates?timeout=40"
    if offset:
        url += "&offset={}".format(offset)
    updates = get_url(url)
    try:
        josn_updates = json.loads(updates)
    except Exception as e:
        print_flush('Error loading updates: {}'.format(str(e)))
        return None
    return josn_updates
    

def get_last_update_id(updates):
    update_ids = []
    for update in updates["result"]:
        update_ids.append(int(update["update_id"]))
    return max(update_ids)

def send_message(text, chatid, reply_markup=None):
    text = urllib.parse.quote_plus(text)
    url = URL + "sendMessage?text={}&chat_id={}&parse_mode=Markdown".format(text, chatid)
    if reply_markup:
        url += "&reply_markup={}".format(reply_markup)
    get_url(url)

def list_unallowed_users():
    global db_conn, db_cursor
    print_flush('list_unallowed_users()')
    try:
        db_cursor.execute('SELECT id, first_name, last_name, nickname, first_contact_timestamp, last_contact_timestamp FROM users WHERE user_allowed = 0')
        unallowed_users = db_cursor.fetchall()
    except sqlite3.Error as e:
        print_flush('Error listing unallowed users: {}'.format(str(e)))
        return json.dumps({"status": "ERROR: {}".format(str(e))})
    return json.dumps({"status": "Unallowed users listed.", "unallowed_users": unallowed_users})

def list_admin_users():
    global db_conn, db_cursor
    print_flush('list_admin_users()')
    try:
        db_cursor.execute('SELECT id, first_name, last_name, nickname, first_contact_timestamp, last_contact_timestamp FROM users WHERE is_admin = 1')
        admin_users = db_cursor.fetchall()
    except sqlite3.Error as e:
        print_flush('Error listing admin users: {}'.format(str(e)))
        return json.dumps({"status": "ERROR: {}".format(str(e))})
    return json.dumps({"status": "Admin users listed.", "admin_users": admin_users})

# Function to disallow a user to chat with the bot
def disallow_chatid_to_chat_with_bot(chatid_to_disallow):
    global db_conn, db_cursor
    print_flush('disallow_chatid_to_chat_with_bot({})'.format(chatid_to_disallow))
    # dont allow the admin user to be disallowed
    if chatid_to_disallow == TELEGRAM_ADMIN_ID:
        return json.dumps({"status": "ERROR: Cannot disallow the admin user."})
    try:
        db_write_lock.acquire()
        db_cursor.execute('UPDATE users SET user_allowed = 0, is_admin = 0 WHERE id = ?', (chatid_to_disallow,))
        db_conn.commit()
        db_write_lock.release()
    except sqlite3.Error as e:
        print_flush('Error disallowing user: {}'.format(str(e)))
        return json.dumps({"status": "ERROR: {}".format(str(e))})
    # clear the user from the active_chats dictionary
    if chatid_to_disallow in active_chats:
        del active_chats[chatid_to_disallow]
    return json.dumps({"status": "User disallowed."})

def allow_chatid_to_chat_with_bot(chatid_to_allow, chatid):
    global db_conn, db_cursor
    print_flush('allow_chatid_to_chat_with_bot({}, {})'.format(chatid_to_allow, chatid))
    try:
        db_write_lock.acquire()
        db_cursor.execute('UPDATE users SET user_allowed = 1 WHERE id = ?', (chatid_to_allow,))
        db_conn.commit()
        db_write_lock.release()
    except sqlite3.Error as e:
        print_flush('Error allowing user: {}'.format(str(e)))
        return json.dumps({"status": "ERROR: {}".format(str(e))})
    # if the user is already in the active_chats dictionary, set the user as an admin
    if chatid_to_allow in active_chats:
        active_chats[chatid_to_allow]['is_admin'] = 1
    return json.dumps({"status": "User allowed."})

def promote_user_to_admin(chatid_to_promote, chatid):
    global db_conn, db_cursor
    print_flush('promote_user_to_admin({}, {})'.format(chatid_to_promote, chatid))
    try:
        db_write_lock.acquire()
        # set the user as an admin
        db_cursor.execute('UPDATE users SET is_admin = 1 WHERE id = ?', (chatid_to_promote,))
        db_conn.commit()
        db_write_lock.release()
    except sqlite3.Error as e:
        print_flush('Error promoting user: {}'.format(str(e)))
        return json.dumps({"status": "ERROR: {}".format(str(e))})
    return json.dumps({"status": "User promoted to admin."})

# function to show the list of expenses stored in the expenses list
def retrieve_expenses(chatid, start_date=None, end_date=None, category=None):
    global db_conn, db_cursor
    print_flush('retrieve_expenses({}, {}, {}, {})'.format(chatid, start_date, end_date, category))
    try:
        if category and start_date and end_date:
            db_cursor.execute('SELECT amount, category, date, description FROM expenses WHERE chatid = ? AND category = ? AND date BETWEEN ? AND ?', (chatid, category, start_date, end_date))
        elif start_date and end_date:
            db_cursor.execute('SELECT amount, category, date, description FROM expenses WHERE chatid = ? AND date BETWEEN ? AND ?', (chatid, start_date, end_date))
        elif category:
            db_cursor.execute('SELECT amount, category, date, description FROM expenses WHERE chatid = ? AND category = ?', (chatid, category))
        else:
            db_cursor.execute('SELECT amount, category, date, description FROM expenses WHERE chatid = ?', (chatid,))
        expenses = db_cursor.fetchall()
    except sqlite3.Error as e:
        print_flush('Error showing expenses: {}'.format(str(e)))
        return json.dumps({"status": "ERROR: {}".format(str(e))})
    return json.dumps({"status": "OK.", "expenses": expenses})

# function to retrieve the list of expense categories stored in the expenses list
def retrieve_expense_categories(chatid):
    global db_conn, db_cursor
    print_flush('retrieve_expense_categories({})'.format(chatid))
    try:
        db_cursor.execute('SELECT DISTINCT category FROM expenses WHERE chatid = ?', (chatid,))
        categories = db_cursor.fetchall()
    except sqlite3.Error as e:
        print_flush('Error retrieving expense categories: {}'.format(str(e)))
        return json.dumps({"status": "ERROR: {}".format(str(e))})
    return json.dumps({"status": "OK.", "categories": categories})

# function to store an expense to the expenses list
def add_expense(amount, category, date, description, chatid):
    global db_conn, db_cursor
    print_flush('add_expense({}, {}, {}, {}, {})'.format(amount, category, date, description, chatid))
    try:
        db_write_lock.acquire()
        db_cursor.execute('INSERT INTO expenses (chatid, amount, category, date, description) VALUES (?, ?, ?, ?, ?)', (chatid, amount, category, date, description))
        db_conn.commit()
        db_write_lock.release()
    except sqlite3.Error as e:
        print_flush('Error storing expense: {}'.format(str(e)))
        return json.dumps({"status": "ERROR: {}".format(str(e))})
    return json.dumps({"status": "Expense stored."})

# function to remove an expense from the expenses list
def remove_expenses(chatid, amount=None, date=None, start_date=None, end_date=None):
    global db_conn, db_cursor
    print_flush('remove_expenses({}, {}, {}, {}, {})'.format(chatid, amount, date, start_date, end_date))
    # check if the amount and date are provided
    if amount and date:
        try:
            db_write_lock.acquire()
            db_cursor.execute('DELETE FROM expenses WHERE chatid = ? AND amount = ? AND date = ?', (chatid, amount, date))
            db_conn.commit()
            db_write_lock.release()
            return json.dumps({"status": "Expense removed."})
        except sqlite3.Error as e:
            print_flush('Error removing expense: {}'.format(str(e)))
            return json.dumps({"status": "ERROR: {}".format(str(e))})
    # check if the start_date and end_date are provided
    elif start_date and end_date:
        try:
            db_write_lock.acquire()
            db_cursor.execute('DELETE FROM expenses WHERE chatid = ? AND date BETWEEN ? AND ?', (chatid, start_date, end_date))
            db_conn.commit()
            db_write_lock.release()
            return json.dumps({"status": "Expenses removed."})
        except sqlite3.Error as e:
            print_flush('Error removing expenses: {}'.format(str(e)))
            return json.dumps({"status": "ERROR: {}".format(str(e))})
    # if neither amount and date nor start_date and end_date are provided, return an error
    else:
        return json.dumps({"status": "ERROR: No amount and date or start_date and end_date provided."})

# function to clear the message history
def clear_message_history(chatid):
    global active_chats
    print_flush('clear_message_history({})'.format(chatid))
    active_chats[chatid]['message_history'] = []
    return json.dumps({"status": "Message history cleared."})

def per_chatid_message_handler(chatid):
        # log that the thread started  
        print_flush('thread started for chatid {}'.format(chatid))

        # loop until the thread event is set
        while True:
            # wait for the event to be set or for 5 seconds to pass
            prompt = active_chats[chatid]['message_queue'].get()
            print_flush('{} - {}'.format(chatid, prompt))
            # Add the datetime to the message history as system message
            # TODO: Only keep one "current date" message in the message history
            active_chats[chatid]['message_history'].append({"role": "system", "content": "Current date is: {}".format(datetime.datetime.now().isoformat())})
            active_chats[chatid]['message_history'].append({"role": "user", "content": prompt})
            try:
                response = openai_client.chat.completions.create(
                    model = active_chats[chatid]['model'],
                    messages = active_chats[chatid]['message_history'],
                    tools = active_chats[chatid]['tools'],
                    tool_choice = "auto"
                )
                # store the completion statistics in the database
                db_write_lock.acquire()
                db_cursor.execute('INSERT INTO completions (chatid, completion_id, completion_created, \
                                  completion_model, completion_response, prompt_tokens, completion_tokens, finish_reason) \
                                  VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (chatid, response.id, response.created, response.model, \
                                    response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens, response.choices[0].finish_reason))
                db_conn.commit()
                db_write_lock.release()
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls
                # check if the response contains a tool call
                if tool_calls:
                    # Step 3: call the function
                    # Note: the JSON response may not always be valid; be sure to handle errors
                    active_chats[chatid]['message_history'].append(response_message)  # extend conversation with assistant's reply
                    # Step 4: send the info for each function call and function response to the model
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_to_call = available_functions[function_name]
                        function_args = json.loads(tool_call.function.arguments)
                        if function_name == "render_dalle_image":
                            function_response = function_to_call(
                                prompt = function_args.get("prompt"),
                                quality = function_args.get("quality"),
                                chatid = chatid,
                            )
                        elif function_name == "generate_text_to_speech":
                            function_response = function_to_call(
                                text = function_args.get("text"),
                                voice = function_args.get("voice"),
                                chatid = chatid,
                            )
                        elif function_name == "gpt_model":
                            function_response = function_to_call(
                                model = function_args.get("model"),
                                chatid = chatid,
                            )
                        elif function_name == "add_thing_to_items_list":
                            function_response = function_to_call(
                                item = function_args.get("item"),
                                owner = function_args.get("owner"),
                                quantity = function_args.get("quantity"),
                                chatid = chatid,
                            )
                        elif function_name == "show_items_list":
                            function_response = function_to_call(
                                owner = function_args.get("owner"),
                                chatid = chatid,
                            )
                        elif function_name == "list_unallowed_users":
                            function_response = function_to_call()
                        elif function_name == "list_admin_users":
                            function_response = function_to_call()
                        elif function_name == "allow_chatid_to_chat_with_bot":
                            function_response = function_to_call(
                                chatid_to_allow = function_args.get("chatid_to_allow"),
                                chatid = chatid,
                            )
                        elif function_name == "promote_user_to_admin":
                            function_response = function_to_call(
                                id_to_promote = function_args.get("chatid_to_promote"),
                                chatid = chatid,
                            )
                        elif function_name == "retrieve_expenses":
                            function_response = function_to_call(
                                category = function_args.get("category"),
                                start_date = function_args.get("start_date"),
                                end_date = function_args.get("end_date"),
                                chatid = chatid,
                            )
                        elif function_name == "retrieve_expense_categories":
                            function_response = function_to_call(
                                chatid = chatid,
                            )
                        elif function_name == "add_expense":
                            function_response = function_to_call(
                                amount = function_args.get("amount"),
                                category = function_args.get("category"),
                                date = function_args.get("date"),
                                description = function_args.get("description"),
                                chatid = chatid,
                            )
                        elif function_name == "remove_expenses":
                            function_response = function_to_call(
                                amount = function_args.get("amount"),
                                date = function_args.get("date"),
                                start_date = function_args.get("start_date"),
                                end_date = function_args.get("end_date"),
                                chatid = chatid,
                            )
                        elif function_name == "clear_message_history":
                            function_response = function_to_call(
                                chatid = chatid,
                            )
                            raise MessageClearedException("Message history cleared.")
                        print_flush('function_response: {}'.format(function_response))
                        # extend conversation with function response
                        active_chats[chatid]['message_history'].append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                            }                            
                        )
                    # get a new response from the model where it can see the function response(s)
                    response = openai_client.chat.completions.create(
                        model = active_chats[chatid]['model'],
                        messages = active_chats[chatid]['message_history'],
                    )
                    # store the completion statistics in the database
                    db_write_lock.acquire()
                    db_cursor.execute('INSERT INTO completions (chatid, completion_id, completion_created, \
                                    completion_model, completion_response, prompt_tokens, completion_tokens) \
                                    VALUES (?, ?, ?, ?, ?, ?, ?)', (chatid, response.id, response.created, response.model, \
                                        response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens))
                    db_conn.commit()
                    db_write_lock.release()
                    # set the response_message to the new response
                    response_message = response.choices[0].message
            except MessageClearedException as e:
                print_flush("MessageException: {}".format(str(e)))
                send_message('Message history has been cleared!', chatid)
                continue
            except Exception as e:
                print_flush("Error generating response: {}".format(str(e)))
                send_message('...failed. Error: {}'.format(str(e)), chatid)
                continue
            active_chats[chatid]['message_history'].append({"role": "assistant", "content": response_message.content})
            send_message(response_message.content, chatid)

def connect_to_database():
    global db_conn, db_cursor
    print_flush('connect_to_database({})'.format(DB_FILE))
    # create the database file if it doesn't exist
    try:
        db_conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    except sqlite3.Error as e:
        raise BaseException('Error connecting to database: {}'.format(str(e)))
    
    # Create the cursor object
    db_cursor = db_conn.cursor()

    # Check the database PRAGMA user_version
    db_cursor.execute('PRAGMA user_version')
    user_version = db_cursor.fetchone()[0]
    print_flush('database user_version: {}'.format(user_version))

    if user_version == 0:
        # Create the "users" table
        db_cursor.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT, nickname TEXT NOT NULL, first_contact TEXT, user_allowed INTEGER);')
        db_conn.commit()

        # Create the "items" table
        db_cursor.execute('CREATE TABLE items (id INTEGER PRIMARY KEY, chatid INTEGER, item TEXT NOT NULL, owner TEXT NOT NULL, quantity INTEGER NOT NULL);')
        db_conn.commit()

        # Create the "images" table
        db_cursor.execute('CREATE TABLE images (id INTEGER PRIMARY KEY, chatid INTEGER NOT NULL, image_filename TEXT NOT NULL, timestamp_created TEXT NOT NULL, timestamp_deleted TEXT, prompt TEXT, revised_prompt TEXT);')

        # Add the admin user to the database if it doesn't exist
        db_cursor.execute('SELECT * FROM users WHERE id = ?', (TELEGRAM_ADMIN_ID,))
        if not db_cursor.fetchone():
            db_cursor.execute('INSERT INTO users (id, nickname, user_allowed) VALUES (?, ?, ?)', (TELEGRAM_ADMIN_ID, 'admin', 1))
            db_conn.commit()
            print_flush('admin user with Telegram id {} added to the database'.format(TELEGRAM_ADMIN_ID))

        # Finally, set the user_version to 1
        user_version = 1
        db_cursor.execute('PRAGMA user_version = {}'.format(user_version))
        db_conn.commit()
        print_flush('Database upgraded to version 1')
    
    if user_version == 1:
        # Alter the "users" table
        db_cursor.execute('ALTER TABLE users RENAME COLUMN first_contact TO first_contact_timestamp')
        db_cursor.execute('ALTER TABLE users ADD COLUMN last_contact_timestamp TEXT')
        db_cursor.execute('ALTER TABLE users ADD COLUMN is_admin INTEGER')
        db_conn.commit()

        # Make the admin user an admin in the database
        db_cursor.execute('UPDATE users SET is_admin = 1 WHERE id = ?', (TELEGRAM_ADMIN_ID,))
        db_conn.commit()
        print_flush('User with Telegram id {} made admin'.format(TELEGRAM_ADMIN_ID))

        # Finally, set the user_version to 2
        user_version = 2
        db_cursor.execute('PRAGMA user_version = {}'.format(user_version))
        db_conn.commit()
        print_flush('Database upgraded to version 2')
    
    if user_version == 2:
        # Add the "expenses" table
        db_cursor.execute('CREATE TABLE expenses (id INTEGER PRIMARY KEY, chatid INTEGER NOT NULL, amount REAL NOT NULL, category TEXT NOT NULL, date TEXT NOT NULL, description TEXT);')
        db_conn.commit()
        print_flush('expenses table added to the database')

        # Finally, set the user_version to 3
        user_version = 3
        db_cursor.execute('PRAGMA user_version = {}'.format(user_version))
        db_conn.commit()
        print_flush('Database upgraded to version 3')

    if user_version == 3:
        # Add fields to the "users" table to store the user's OpenAI token statistics
        db_cursor.execute('ALTER TABLE users ADD COLUMN total_completion_tokens INTEGER')
        db_cursor.execute('ALTER TABLE users ADD COLUMN total_prompt_tokens INTEGER')
        db_cursor.execute('ALTER TABLE users ADD COLUMN total_images INTEGER')
        db_conn.commit()
        print_flush('fields added to the users table to store the user\'s OpenAI token statistics')

        # Add the "completions" table to store the completions
        db_cursor.execute('CREATE TABLE completions (id INTEGER PRIMARY KEY, chatid INTEGER NOT NULL, \
                          completion_id TEXT NOT NULL, completion_created TEXT NOT NULL, \
                          completion_model TEXT NOT NULL, completion_response TEXT NOT NULL, \
                          prompt_tokens TEXT NOT NULL, completion_tokens TEXT NOT NULL);')
        db_conn.commit()
        print_flush('completions table added to the database')

        # Finally, set the user_version to 4
        user_version = 4
        db_cursor.execute('PRAGMA user_version = {}'.format(user_version))
        db_conn.commit()
        print_flush('Database upgraded to version 4')

    if user_version == 4:
        # Add the finish reason to the completions table
        db_cursor.execute('ALTER TABLE completions ADD COLUMN finish_reason TEXT')
        # Allow empty completion_response in the completions table
        db_cursor.execute('ALTER TABLE completions ALTER COLUMN completion_response TEXT')
        db_conn.commit()
        print_flush('finish reason added to the completions table, completion_response can be empty')

        # Finally, set the user_version to 5
        user_version = 5
        db_cursor.execute('PRAGMA user_version = {}'.format(user_version))
        db_conn.commit()
        print_flush('Database upgraded to version 5')

    print_flush('database opened')

# Function to transcribe the voice message to text.
# This function is run in a separate thread.
# The function downloads the voice message, transcribes it to text, and adds the text to the message queue.
def extract_text_from_voice_message(voice_message, chatid):
    print_flush('extract_text_from_voice_message({}, {})'.format(voice_message, chatid))
    # create the TRANSCRIPTION_FILE_PATH if it doesn't exist
    try:
        if not os.path.exists(TRANSCRIPTION_FILE_PATH):
            os.makedirs(TRANSCRIPTION_FILE_PATH)
    except Exception as e:
        print_flush('Error creating the transcription file path: {}'.format(str(e)))
        return
    # download the voice message from Telegram
    voice_file_path = TRANSCRIPTION_FILE_PATH + "/voice-{}.ogg".format(datetime.datetime.now().timestamp())
    try:
        response = requests.post(TELEGRAM_GETFILE, data={'file_id': voice_message['file_id']}, timeout=60)
    except Exception as e:
        print_flush('Error preparing the file for download: {}'.format(str(e)))
        return
    if response.status_code != 200:
        print_flush('Error preparing the file for download: {}'.format(response.status_code))
        return
    # turn the response into a JSON object
    try:
        response = response.json()
    except Exception as e:
        print_flush('Error turning the response into a JSON object: {}'.format(str(e)))
        return
    if 'file_path' not in response['result']:
        print_flush('Response did not contain the key "file_path"')
        return
    # download the voice message
    voice_file_url = TELEGRAM_FILE + response['result']['file_path']
    try:
        response = requests.get(voice_file_url)
        with open(voice_file_path, "wb") as voice_file:
            voice_file.write(response.content)
        pass
    except Exception as e:
        print_flush('Error downloading the voice message: {}'.format(str(e)))
        return
    
    # Do the transcription using OpenAI's API
    try:
        with open(voice_file_path, "rb") as voice_file:
            transcript = openai_client.audio.transcriptions.create(
                model = "whisper-1",
                file = voice_file,
                response_format = "json",
            )
    except Exception as e:
        print_flush('Error transcribing the voice message to text: {}'.format(str(e)))
        return
    
    # Now delete the voice message file
    try:
        os.remove(voice_file_path)
    except Exception as e:
        print_flush('Error deleting the voice message file: {}'.format(str(e)))
        
    # Finally, add the transcribed text to the message queue
    if transcript.text:
        active_chats[chatid]['message_queue'].put(transcript.text)
    else:
        print_flush('Transcription did not contain the key "text"')
        send_message('Sorry, I could not transcribe the voice message.', chatid)

# Funtion to perform regular maintenance tasks.
# Tasks:
# - Remove old images from the filesystem. We store 10 images per chatid and remove the oldest ones.
# - Remove old messages from the message history. Clear the message history 24h after the last message received.
# TODO: Implement message history pruning here (e.g. keep only the last 10 messages, summarize the rest)
def maintenance_tasks():
    global db_conn, db_cursor
    while True:
        # retrieve list of images from database that meets following criteria:
        # - timestamp_deleted is NULL
        db_cursor.execute('SELECT id, chatid, image_filename FROM images WHERE timestamp_deleted IS NULL')
        images = db_cursor.fetchall()
        # For each chatid, if there are more than 10 images, remove the oldest ones.
        # Strategy: sort the images by id and remove the smallest ones
        for chatid in set([image[1] for image in images]):
            chatid_images = [image for image in images if image[1] == chatid]
            if len(chatid_images) > IMAGES_TO_KEEP_PER_CHATID:
                print_flush('MAINTENANCE({}) - images - User has more than {} iamges'.format(chatid, IMAGES_TO_KEEP_PER_CHATID))
                # sort the images by id
                chatid_images.sort(key=lambda x: x[0])
                # remove the oldest images
                for image in chatid_images[:len(chatid_images) - IMAGES_TO_KEEP_PER_CHATID]:
                    # remove the image from the filesystem
                    print_flush('MAINTENANCE({}) - images - Removing image from the filesystem: {}'.format(chatid, image[2]))
                    image_to_remove = os.path.join(IMAGE_FILE_PATH, str(chatid), image[2])
                    try:
                        os.remove(image_to_remove)
                    except Exception as e:
                        print_flush('MAINTENANCE({}) - images - Error removing "{}" from the filesystem: {}'.format(chatid, image_to_remove, str(e)))
                        continue
                    # mark the image as deleted in the database
                    try:
                        db_write_lock.acquire()
                        db_cursor.execute('UPDATE images SET timestamp_deleted = ? WHERE id = ?', (datetime.datetime.now().isoformat(), image[0]))
                        db_conn.commit()
                        db_write_lock.release()
                    except sqlite3.Error as e:
                        print_flush('MAINTENANCE({}) - images - Error marking image as deleted in the database: {}'.format(chatid, str(e)))

        # For all active chats with message history, clear the message history if the last_contact_timestamp is older than MESSAGE_HISTORY_TIMEOUT_MINUTES minutes
        for chatid in active_chats:
            if active_chats[chatid]['message_history']:
                if (active_chats[chatid]['last_contact_timestamp'] < (datetime.datetime.now() - datetime.timedelta(minutes=MESSAGE_HISTORY_TIMEOUT_MINUTES)).isoformat()):
                    print_flush('MAINTENANCE({}) - message history - clearing message history due to inactivity'.format(chatid))
                    active_chats[chatid]['message_history'] = []
    
        # sleep for MATINTENANCE_TASKS_INTERVAL seconds
        time.sleep(MATINTENANCE_TASKS_INTERVAL)

def main():
    global db_conn, db_cursor
    try:
        connect_to_database()
    except BaseException as e:
        print_flush('Error connecting to database: {}'.format(str(e)))
        return
    # start a thread for regular maintenance tasks
    # TODO: Close the thread when the program exits
    maintenance_thread = threading.Thread(target=maintenance_tasks)
    maintenance_thread.start()

    last_update_id = None
    while True:
        updates = get_updates(last_update_id)
        # cehck if updates is not NULL and contains the key 'result'
        if not updates:
            print_flush('updates is NULL')
            continue
        if not 'result' in updates:
            print_flush('updates did not contain the key "result"')
            continue
        if len(updates['result']) > 0:
            last_update_id = get_last_update_id(updates) + 1
            for update in updates['result']:
                if not 'message' in update:
                    print_flush('Update {} did not contain a message. TODO: Implement other types of updates.'.format(update["update_id"]))
                    continue
                if not 'chat' in update['message']:
                    print_flush('update {} did not contain chat'.format(str(update)))
                    continue
                if not 'id' in update['message']['chat']:
                    print_flush('update {} did not contain chatid'.format(str(update)))
                    continue
                chatid = update['message']['chat']['id']

                # Check if the table users in the database contains the chatid & the user is allowed
                db_cursor.execute('SELECT * FROM users WHERE id = ? AND user_allowed = 1', (chatid,))
                if not db_cursor.fetchone():
                    print_flush('discarding message from {}'.format(chatid))
                    db_cursor.execute('SELECT * FROM users WHERE id = ?', (chatid,))
                    if not db_cursor.fetchone():
                        print_flush('adding user to database: {}'.format(chatid))
                        nickname = update['message']['from']['username']
                        if not nickname:
                            nickname = update['message']['from']['id']
                        first_name = update['message']['from']['first_name']
                        last_name = update['message']['from']['last_name']
                        db_write_lock.acquire()
                        db_cursor.execute('INSERT INTO users (id, nickname, first_name, last_name, user_allowed, first_contact_timestamp) \
                                          VALUES (?, ?, ?, ?, ?, ?)', (chatid, nickname, first_name, last_name, 0, datetime.datetime.now().isoformat()))
                        db_conn.commit()
                        db_write_lock.release()
                else:
                    # Create a queue for the chatid if it doesn't exist
                    if chatid not in active_chats:
                        print_flush('creating queue for chatid {}'.format(chatid))
                        # Create a dictionary for the chatid
                        active_chats[chatid] = dict()
                        # Create an array to store messages history for the chatid
                        active_chats[chatid]['message_history'] = []
                        # Initialize the queue for the chatid
                        active_chats[chatid]['message_queue'] = queue.Queue()
                        # Set the model to use for the chatid
                        active_chats[chatid]['model'] = openai_default_chat_model
                        # Store whether the user is an admin
                        db_cursor.execute('SELECT * FROM users WHERE id = ? AND is_admin = 1', (chatid,))
                        active_chats[chatid]['is_admin'] = db_cursor.fetchone()

                        # Add the tools to the chatid
                        active_chats[chatid]['tools'] = [
                            {
                                "type": "function",
                                "function": {
                                    "name": "render_dalle_image",
                                    "description": "Generate an image using OpenAI's DALL-E model",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "prompt": {
                                                "type": "string",
                                                "description": "The prompt to use for generating the image",
                                            },
                                            "quality": {
                                                "type": "string",
                                                "enum": ["standard", "hd"],
                                                "description": "The quality of the image to generate, can be either standard or hd",
                                            },
                                        },
                                        "required": ["prompt", "quality"],
                                    },
                                },
                            },
                            {
                                "type": "function",
                                "function": {
                                    "name": "generate_text_to_speech",
                                    "description": "Generate text to speech using OpenAI's TTS model. The available voices are alloy, echo, fable, onyx, nova, and shimmer. The default voice is onyx.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "description": "The text to use for the text-to-speech (TTS) generation",
                                            },
                                            "voice": {
                                                "type": "string",
                                                "enum": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                                                "description": "The voice to use for the text-to-speech (TTS) generation. Available voices are alloy, echo, fable, onyx, nova, and shimmer. The default voice is onyx.",
                                            },
                                        },
                                        "required": ["text", "voice"],
                                    },
                                },
                            },
                            {
                                "type": "function",
                                "function": {
                                    "name": "add_thing_to_items_list",
                                    "description": "Stores an item in the items list. Needs the name of the item, the owner of the the item and the quantity of the item owned.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "item": {
                                                "type": "string",
                                                "description": "The item to add to the items list",
                                            },
                                            "owner": {
                                                "type": "string",
                                                "description": "The owner of the item",
                                            },
                                            "quantity": {
                                                "type": "integer",
                                                "description": "The quantity of the item owned by the owner",
                                            },
                                        },
                                        "required": ["item", "owner", "quantity"],
                                    },
                                },
                            },
                            {
                                "type": "function",
                                "function": {
                                    "name": "show_items_list",
                                    "description": "Shows the list of items stored in the items list. Optionally provide the owner of the items to show only the items owned by the owner.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "owner": {
                                                "type": "string",
                                                "description": "The name of the owner of the items to show. If provided, only the items owned by the owner will be shown.",
                                            },
                                        },
                                    },
                                },
                            },
                            # function to add an expense to the expenses list. This function needs the dollar amount, the category, and the date of the expense
                            {
                                "type": "function",
                                "function": {
                                    "name": "add_expense",
                                    "description": "Add an expense to the expenses list. Call this when the user explicitly mentions that he/she spent money. This function needs the dollar amount, the category, the description and the date of the expense. Try to use an existing category, only use a new category if need be. Existing categories can be found with the function retrieve_expense_categories.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "amount": {
                                                "type": "number",
                                                "description": "The dollar amount of the expense",
                                            },
                                            "category": {
                                                "type": "string",
                                                "description": "The category of the expense. This must be a single word. Try to use an existing category, only use a new category if need be. Existing categories can be found with the function retrieve_expense_categories.",
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "A short description of the expense",
                                            },
                                            "date": {
                                                "type": "string",
                                                "description": "The date of the expense. Must be in the format YYYY-MM-DD.",
                                            },
                                        },
                                        "required": ["amount", "category", "description", "date"],
                                    },
                                },                                
                            },
                            # function to remove an expense from the expenses list
                            {
                                "type": "function",
                                "function": {
                                    "name": "remove_expenses",
                                    "description": "Removes expenses from the expenses list. Needs either the date and dollar amount, or the date range of the expenses to remove. Be careful with the date range, as it will remove all expenses in that range. When the user refers to the expenses of the last year / week / month, use the range of the last year / week / month. For example, if today is 2022-01-15, and the user says 'remove all expenses of last month', use the range 2021-12-01 to 2021-12-31, if the user says 'remove all expenses of last year', use the range 2021-01-01 to 2021-12-31.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "amount": {
                                                "type": "number",
                                                "description": "The dollar amount of the expense to remove",
                                            },
                                            "date": {
                                                "type": "string",
                                                "description": "The date of the expense to remove. Must be in the format YYYY-MM-DD.",
                                            },
                                            "start_date": {
                                                "type": "string",
                                                "description": "The start date of the expenses to remove. Must be in the format YYYY-MM-DD. Must be provided if end_date is provided.",
                                            },
                                            "end_date": {
                                                "type": "string",
                                                "description": "The end date of the expenses to remove. Must be in the format YYYY-MM-DD. Must be provided if start_date is provided.",
                                            },
                                        },
                                    },
                                },
                            },
                            # function to retrieve the list of expenses stored in the expenses list
                            {
                                "type": "function",
                                "function": {
                                    "name": "retrieve_expenses",
                                    "description": "Retrieves the list of expenses stored in the expenses list. The category can optionally be passed as an argument to only show only the expenses in that category. The start date and end date can optionally be passed as an argument to only show the expenses in that date range. If no parameters are provided, all the expenses will be shown.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "category": {
                                                "type": "string",
                                                "description": "The category of the expenses to retrieve. If provided, only the expenses in the category will be retrieved.",
                                            },
                                            "start_date": {
                                                "type": "string",
                                                "description": "The start date of the expenses to retrieve. If provided, only the expenses from the start date onward will be retrieved. Must be provided if end_date is provided.",
                                            },
                                            "end_date": {
                                                "type": "string",
                                                "description": "The end date of the expenses to retrieve. If provided, only the expenses up to the end date will be retrieved. Must be provided if start_date is provided.",
                                            },
                                        },
                                    },
                                },
                            },
                            # function to retrieve the list of expense categories stored in the expenses list
                            {
                                "type": "function",
                                "function": {
                                    "name": "retrieve_expense_categories",
                                    "description": "Retrieves the list of expense categories stored in the expenses list.",
                                },
                            },
                            # function to clear the message history
                            {
                                "type": "function",
                                "function": {
                                    "name": "clear_message_history",
                                    "description": "Clears the message history for the chatid.",
                                },
                            }
                        ]

                        # Add the admin tools if the user is an admin. These are:
                        # - list unallowed users (chatid, first_name, last_name, nickname, first_contact_timestamp, last_contact_timestamp)
                        # - allow user by chatid
                        # TODO: Add more admin tools
                        if active_chats[chatid]['is_admin']:
                            active_chats[chatid]['tools'].extend([
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "list_unallowed_users",
                                        "description": "Lists the unallowed users in the database. Returns the chatid, first_name, last_name, nickname, first_contact_timestamp, and last_contact_timestamp of the unallowed users.",
                                    },
                                },
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "list_admin_users",
                                        "description": "Lists the admin users in the database. Returns the chatid, first_name, last_name, nickname, first_contact_timestamp, and last_contact_timestamp of the admin users.",
                                    },
                                },
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "allow_chatid_to_chat_with_bot",
                                        "description": "Allows a user to use the chatbot. Needs the chatid of the user to allow.",
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "chatid_to_allow": {
                                                    "type": "string",
                                                    "description": "The chatid of the user to allow to use the chatbot.",
                                                },
                                            },
                                            "required": ["chatid_to_allow"],
                                        },
                                    },
                                },
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "promote_user_to_admin",
                                        "description": "Promotes a user to admin. Needs the chatid of the user to promote.",
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "chatid_to_promote": {
                                                    "type": "string",
                                                    "description": "The chatid of the user to promote to admin.",
                                                },
                                            },
                                            "required": ["chatid_to_promote"],
                                        },
                                    },
                                },
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "disallow_chatid_to_chat_with_bot",
                                        "description": "Disallows / bans / removes a user to use the chatbot. Needs the chatid of the user to disallow.",
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "chatid_to_disallow": {
                                                    "type": "string",
                                                    "description": "The chatid of the user to disallow to use the chatbot.",
                                                },
                                            },
                                            "required": ["chatid_to_disallow"],
                                        },
                                    },
                                },
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "gpt_model",
                                        "description": "Set the GPT flavor / model to use for the chat. If called without parameter 'model', gpt_model it will return the currently active model. The only vailable models are gpt-3.5-turbo-0125 and gpt-4-turbo-preview. The default model is {}.".format(openai_default_chat_model),
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "model": {
                                                    "type": "string",
                                                    "description": "The GPT flavor / model to use for the chat. The only available models are gpt-3.5-turbo-0125 and gpt-4-turbo-preview. When using this parameter, it is important that only the available models can be selected. The default model is {}.".format(openai_default_chat_model),
                                                },
                                            },
                                        },
                                    },
                                },
                            ])

                        # Start a thread to handle messages for the chatid and store it as a value in the active_chats dictionary
                        active_chats[chatid]['thread'] = threading.Thread(target=per_chatid_message_handler, args=(chatid,))
                        active_chats[chatid]['thread'].start()
                
                    # Store the timestamp of the last contact with the user to the database and into the active_chats dictionary
                    db_write_lock.acquire()
                    db_cursor.execute('UPDATE users SET last_contact_timestamp = ? WHERE id = ?', (datetime.datetime.now().isoformat(), chatid))
                    db_conn.commit()
                    db_write_lock.release()
                    active_chats[chatid]['last_contact_timestamp'] = datetime.datetime.now().isoformat()

                    print_flush('received message from allowed user {}'.format(chatid))

                    # check what type of message it is    
                    if 'text' in update['message']: # text message
                        # add the message to the queue
                        active_chats[chatid]['message_queue'].put(update['message']['text'])
                        continue
                    elif 'photo' in update['message']: # photo message
                        print_flush('TODO: implement photo message handling')
                        continue
                    elif 'voice' in update['message']: # voice message
                        # start a thread to handle the voice message
                        voice_thread = threading.Thread(target=extract_text_from_voice_message, args=(update['message']['voice'], chatid))
                        voice_thread.start()
                        continue
                    elif 'video' in update['message']: # video message
                        print_flush('TODO: implement video message handling')
                        continue
                    elif 'document' in update['message']: # document message
                        print_flush('TODO: implement document message handling')
                        continue
                    else:
                        print_flush('No supported message type found in the update: {}'.format(update))
                        continue

# array with available functions
available_functions = {
    "render_dalle_image": render_dalle_image,
    "generate_text_to_speech": generate_text_to_speech,
    "add_thing_to_items_list": add_thing_to_items_list,
    "show_items_list": show_items_list,
    "retrieve_expenses": retrieve_expenses,
    "retrieve_expense_categories": retrieve_expense_categories,
    "add_expense": add_expense,
    "remove_expenses": remove_expenses,
    "clear_message_history": clear_message_history,
    "list_unallowed_users": list_unallowed_users,
    "list_admin_users": list_admin_users,
    "allow_chatid_to_chat_with_bot": allow_chatid_to_chat_with_bot,
    "disallow_chatid_to_chat_with_bot": disallow_chatid_to_chat_with_bot,
    "promote_user_to_admin": promote_user_to_admin,
    "gpt_model": gpt_model,
}

if __name__ == '__main__':
    main()
