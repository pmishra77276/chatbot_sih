import random
import json
import pickle
import numpy as np
import nltk
import re
from datetime import datetime,timedelta
import spacy

import os
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents4.json').read())

words = pickle.load(open('words5.pkl', 'rb'))
classes = pickle.load(open('classes5.pkl', 'rb'))
model = load_model('test12.h5')

museum_list = {
    "Delhi Science Center": "National Science Centre Delhi",
    "Delhi Science Museum": "National Science Centre Delhi",
    "National science museum":'National Science Centre Delhi',
    "National science center":'National Science Centre Delhi',
    "Delhi museum":'National Science Centre Delhi'
    }
def update(d):
    ticketPrices = {
        'General Entry': 70,
        'General Entry (Group >25)': 60,
        'General Entry (BPL Card)': 20,
        'Students Entry (School Group)': 25,
        'Students Entry (Govt/MCD School)': 10,
        '3D Film': 40,
        'SDL/Taramandal (Adult)': 20,
        'SDL/Taramandal (Children)': 20,
        'SOS Entry (Adult)': 50,
        'Holoshow Entry (Adult)': 40,
        'Fantasy Ride': 80,
        'Package (All Inclusive)':250
    }
    print("What you want to update :: ")
    print("1. For Name\n2. For location\n3. For Ticekt type\n 4. For Date")
    while(True):
        ch=int(input("please give your choice (0 to exit) :: "))
        if(ch==1):
            d['name']=input("Please give me your name :: ")
        elif(ch==2):
            d['location']=input("Please tell me about location :: ")
        elif(ch==3):
            print(ticketPrices)
            print()
            li=[]
            print("To exit type 'exit'")
            while(True):
                a=""
                while True:
                    b=input("Give me name of tickets :: ")
                    if(b=='exit'):
                        a=b
                        break
                    elif(b.title() not in list(ticketPrices.keys())):
                        print("Give correct information")
                    else:
                        a=b.title()
                        break
                # print(a)
                if(a.lower()=='exit'):
                    break
                num=int(input(f"Number of peoples booking {a} tickets :: "))
                val=[a,num]
                if a!="":
                    li.append(val)
                else:
                    li=None
            if(li!=None):
                d['ticket_type']=li
        elif(ch==4):
            me=""
            while True:
                mes=input("Please tell when you will visit museum dd-mm-yyyy :: ")
                date = validate_date1(mes)
                if(date!=False):
                    # print(date)
                    me=mes
                    break
                else:
                    print("Give date in correct format (It should not be older than today)")
            d['visiting_date']=me
        elif(ch==0):
            break
            
            
def extract_museum_from_list(text, museums):
    text = text.lower()
    for museum in museums:
        if museum.lower() in text:
            return museums[museum]
    return None

def validate_date1(date_str):
    try:
        # Parse the input date string in the dd/mm/yyyy format
        input_date = datetime.strptime(date_str, '%d-%m-%Y')
        # Get today's date without the time part
        today = datetime.today().date()
        # Check if the input date is in the past
        if input_date.date() < today:
            return False  # Invalid if it's older than today's date
        return True  # Valid if the date is today or in the future
    except ValueError:
        return False  # Invalid if the format doesn't match

def validate_date(day, month, year):
    try:
        if None in (day, month, year):
            return None
        date = datetime(year, month, day)
        if date < datetime.now().replace(hour=0, minute=0, second=0, microsecond=0):
            return None
        return date
    except ValueError:
        return None

def month_name_to_number(month_name):
    month_name = month_name.lower()
    month_map = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5, 'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12
    }
    return month_map.get(month_name, None)

def extract_date_from_keywords(text):
    if re.search(r'\bphele\b|\bbefore\b', text, re.IGNORECASE):
        return None
    if re.search(r'\baaj se\b|\bfrom today\b', text, re.IGNORECASE):
        match = re.search(r'(\d+)\s*(?:days)?\s*(?:aaj se|from today)', text, re.IGNORECASE)
        if not match:
            match = re.search(r'(?:aaj se|from today)\s*(\d+)\s*(?:days)?', text, re.IGNORECASE)
        if match:
            days_to_add = int(match.group(1))
            return datetime.now() + timedelta(days=days_to_add)
    elif re.search(r'\btoday\b|\baaj\b', text, re.IGNORECASE):
        return datetime.now()
    elif re.search(r'\bday after tomorrow\b|\bperso\b', text, re.IGNORECASE):
        return datetime.now() + timedelta(days=2)
    elif re.search(r'\btomorrow\b|\bkal\b', text, re.IGNORECASE):
        return datetime.now() + timedelta(days=1)
    return None

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)
def predict_class (sentence):
    bow = bag_of_words (sentence)
    
    res = model.predict(np.array([bow]))[0]
    
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    # print(results)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    return result

def validate(d):
    for i in d:
        if(d[i]==None):
            return [False,i]
    if(len(d['ticket_type'])==0):
        return [False,'ticket_type']
    return [True]

def main():
    d={"name":None,"location":None,"ticket_type":None,"visiting_date":None}
    ticketPrices = {
        'General Entry': 70,
        'General Entry (Group >25)': 60,
        'General Entry (BPL Card)': 20,
        'Students Entry (School Group)': 25,
        'Students Entry (Govt/MCD School)': 10,
        '3D Film': 40,
        'SDL/Taramandal (Adult)': 20,
        'SDL/Taramandal (Children)': 20,
        'SOS Entry (Adult)': 50,
        'Holoshow Entry (Adult)': 40,
        'Fantasy Ride': 80,
        'Package (All Inclusive)':250
    }
    museum_list=[
        "National Science Centre Delhi",
        "National Railway Museum",
        "Victoria Memorial Hall"
        "Allahabad Museum",
        "Salar Jung Museum"
    ]
    booking_flag1=0
    update_flag=0
    while(True):
        
        message = input("Enter messagae :")
        ints = predict_class (message)
        print(ints)
        flag=0
        for i in range(len(ints)):
            if(float(ints[i]["probability"])>=0.5):
                res = get_response (ints, intents)
                print(res)
                if(ints[i]['intent']=='book_ticket' or ints[i]['intent']=='parchi_kaatna'):
                    booking_flag1=1
                    user_text=message
                    date_pattern = (
                        r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b|'
                        r'\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b|'
                        r'\b(\d{1,2})(?:st|nd|rd|th)? (\w{3,9})(?: (\d{4}))?\b|'
                        r'\b(\d{1,2}) (\w{3,9})(?:,? (\d{4}))?\b'
                    )
                    date = extract_date_from_keywords(user_text)
                    if not date:
                        date_match = re.search(date_pattern, user_text)
                        current_year = datetime.now().year
                        if date_match:
                            if date_match.group(1):
                                day, month, year = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
                                if year < 100:
                                    year += 2000
                                date = validate_date(day, month, year)
                            elif date_match.group(4):
                                day, month, year = int(date_match.group(4)), int(date_match.group(5)), int(date_match.group(6))
                                if year < 100:
                                    year += 2000
                                date = validate_date(day, month, year)
                            elif date_match.group(7):
                                day = int(date_match.group(7))
                                month = month_name_to_number(date_match.group(8).lower())
                                year = int(date_match.group(9)) if date_match.group(9) else current_year
                                date = validate_date(day, month, year)
                            elif date_match.group(10):
                                day = int(date_match.group(11))
                                month = month_name_to_number(date_match.group(10).lower())
                                year = int(date_match.group(12)) if date_match.group(12) else current_year
                                date = validate_date(day, month, year)
                    
                    museum = extract_museum_from_list(user_text, museum_list)
                    
                    if not museum:
                        museum=None
                    if date:
                        try:
                            if(update_flag==0):
                                d['visiting_date']=date.strftime('%d-%m-%Y')
                                print(f"Date: {date.strftime('%d-%m-%Y')}")
                        except AttributeError:
                            print("Invalid date")
                    else:
                        print("Invalid date")
                    if(museum and update_flag==0):
                        d['location']=museum
                        print(f"Museum: {museum if museum else 'Not found'}")
                    
                    if(booking_flag1==1 and validate(d)[0]==False):
                        while(True):
                            info_req=validate(d)[1]
                            mes=""
                            if(info_req=='name'):
                                mes=input(f"Please provide me your {info_req} :: ")
                                d[info_req]=mes
                            elif(info_req=='ticket_type'):
                                print(ticketPrices)
                                print()
                                li=[]
                                print("To exit type 'exit'")
                                while(True):
                                    a=""
                                    while True:
                                        b=input("Give me name of tickets :: ")
                                        if(b=='exit'):
                                            a=b
                                            break
                                        elif(b.title() not in list(ticketPrices.keys())):
                                            print("Give correct information")
                                        else:
                                            a=b.title()
                                            break
                                    # print(a)
                                    if(a.lower()=='exit'):
                                        break
                                    num=int(input(f"Number of peoples booking {a} tickets :: "))
                                    val=[a,num]
                                    if a!="":
                                        li.append(val)
                                    else:
                                        li=None
                                if(li!=None):
                                    d['ticket_type']=li
                            elif(info_req=='visiting_date'):
                                me=""
                                while True:
                                    mes=input("Please tell when you will visit museum dd-mm-yyyy :: ")
                                
                                
                                    date = validate_date1(mes)
                                    if(date!=False):
                                        # print(date)
                                        me=mes
                                        break
                                    else:
                                        print("Give date in correct format (It should not be older than today)")
                                d['visiting_date']=me
                            elif info_req=='location':
                                print(museum_list)
                                while(True):
                                    mes=input("please tell me name of museum :: ")
                                    if(mes.title() in museum_list):
                                        d['location']=mes
                                        break
                                    else:
                                        print("Please give correct info")
                            if(validate(d)[0]==True or "cancel" in mes):
                                break
                        print(d)
                        if(update_flag!=1):
                            while(True):
                                print("Do you want to update (y/n)")
                                ch=input()
                                if(ch.lower()=='y'):
                                    update(d)
                                    print("Updated info :: ",d)
                                else:
                                    # update_flag=1
                                    print("booking done")
                                    
                                    
                                    break
                        else:
                            print("Booking already done")
                    else:
                        print(d)
                        if(update_flag!=1):
                            while(True):
                                print("Do you want to update (y/n)")
                                ch=input()
                                if(ch.lower()=='y' and update_flag!=1):
                                    # update_flag=1
                                    update(d)
                                    print("Updated info :: ",d)
                                else:
                                    # update_flag=1
                                    print("please confirm your bookings")
                                    break
                        else:
                            print("Booking already done")
                        # print("Booking already done")
                flag=1
                # res = get_response (ints, intents)
                # print(res)
                if(ints[i]['intent']=='confirm_booking' and validate(d)[0]!=False):
                    update_flag=1
                    print("Pranjal pass dictionary to rakshit")
                    #pranjal pass dictionary d to rakshit here
                break
        if(flag==0):
            print("For this query I don't know the answer please contact (033) 23576008.")
        
        if "bye" in message.lower():
            break
if __name__ == "__main__":
    main()