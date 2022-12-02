import json
import boto3
import email
import os
import re
import sys
import numpy as np
from hashlib import md5
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans
    
def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1. 
    return results

def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data    
    
def one_hot(text, n, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=' '):
    return hashing_trick(text, n,hash_function='md5',filters=filters,lower=lower,split=split)

def text_to_word_sequence(text,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True, split=" "):
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

def hashing_trick(text, n,hash_function=None,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=' '):
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,filters=filters,lower=lower,split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]
    
def send_response_email(send_date_converted, sender_email, subject, mail_body, prediction, score):
    score = score * 100;
    response_mail = "We received your email sent at %s with the subject \"%s\".\n\nHere is a 240 character sample of the email body:\n\n%s\n\nThe email was categorized as %s with a %.2f%% confidence." % (send_date_converted, subject, mail_body[0:240], prediction, score);
    # print("Sending [%s] mail to [%s]" % (response_mail, sender_email));
    client = boto3.client('ses');
    response = client.send_email(Source='test@cloudding.me', 
        Destination={'ToAddresses': [sender_email],},
        Message={
            'Subject': {
                'Data': subject,
                'Charset': 'utf-8'
            },
            'Body': {
                'Text': {
                    'Data': response_mail,
                    'Charset': 'utf-8'
                }
            }
        }
    );
    # print(response);

def lambda_handler(event, context):
    # TODO implement
    
    print("event :", event)
    s3_bucket = event['Records'][0]['s3']['bucket']['name']
    # s3_bucket = "cloud-hw3-s1"
    s3_key = event['Records'][0]['s3']['object']['key']
    # s3_key = '4kjfmpugk5hnov9g7mvrngon23oki025arakeu81'
    
    client = boto3.client('s3')
    data = client.get_object(Bucket=s3_bucket, Key=s3_key)
    contents = data['Body'].read()
    print("contents: ", contents)
    msg = email.message_from_bytes(contents)
    
    send_date = re.search('Date: (.*) *[\r\n]', contents.decode('utf8')).group(1).strip()
    sender_email = re.search('From: (.*) *[\r\n]',  contents.decode('utf8')).group(1).strip()
    check1 = re.search('[^<]* *<(.*@.*)>', sender_email)
    if check1:
        sender_email = check1.group(1)

    ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
    # ENDPOINT_NAME = 'sms-spam-classifier-mxnet-2022-11-28-20-31-15-720'
    runtime= boto3.client('runtime.sagemaker')   
    
    payload = ""
    
    if msg.is_multipart():
        print("multi part")
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))

        # skip any text/plain (txt) attachments
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                payload = part.get_payload(decode=True)  # decode
                print("multi part", payload)
                break
    else:
        #print("msg payload is = ", msg.get_payload())
        payload = msg.get_payload()
        
    
    #print("payload is ", payload.decode("utf-8"))
    payload = payload.decode("utf-8")
    payload = payload.replace('\r\n',' ').strip()
    
    payloadtext = payload
    
    vocabulary_length = 9013
    test_messages = [payload]
    #test_messages = ["FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop"]
    one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    payload = json.dumps(encoded_test_messages.tolist())
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,ContentType='application/json',Body=payload)
    logger.debug(f"Endpoint response is: {response}")
    
    response_body = response['Body'].read().decode('utf-8')
    result = json.loads(response_body)

    logger.debug(f"Result is: {result}")
    pred = int(result['predicted_label'][0][0])
    if pred == 1:
        CLASSIFICATION = "SPAM"
    elif pred == 0:
        CLASSIFICATION = "NOT SPAM"
    # CLASSIFICATION_CONFIDENCE_SCORE = str(float(result['predicted_probability'][0][0]) * 100)
    CLASSIFICATION_CONFIDENCE_SCORE = float(result['predicted_probability'][0][0])
    
    logger.debug(f"------ MSG -> {msg}")
    SENDER = "XXXXXXXXXXXXX"
    RECIPIENT = msg['From']
    logger.debug(f"RECIPIENT is: {RECIPIENT}")
    EMAIL_RECEIVE_DATE = msg["Date"]
    logger.debug(f"EMAIL_RECEIVE_DATE is: {EMAIL_RECEIVE_DATE}")
    EMAIL_SUBJECT = msg["Subject"]
    payloadtext = payloadtext[:240]
    EMAIL_BODY = payloadtext
    AWS_REGION = "us-east-1"

    # The email to send.
    SUBJECT = "Homework Assignment 3"

    CHARSET = "UTF-8"
    send_response_email(EMAIL_RECEIVE_DATE, RECIPIENT, EMAIL_SUBJECT, EMAIL_BODY, CLASSIFICATION, CLASSIFICATION_CONFIDENCE_SCORE)
    return {
        'statusCode': 200,
        'body': json.dumps('Success to send email')
    }