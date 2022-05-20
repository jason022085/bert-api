# -*- encoding: utf8 -*-
import json
import sys
import traceback
import logging
import hashlib

from flask import session, request, abort


def auto_write_logging_api(f):
    def wrapped(self, *args, **kwargs):
        class_nm = self.__class__.__name__
        owner = session.get('user_id')
        level, ip_address = "INFO", request.remote_addr

        payload = request.json

        try:
            result = f(self, *args, **kwargs)
            if isinstance(result, dict):
                str_result = json.dumps(result)
                result_state = result.get("result")
                if result_state == 1:
                    level = "WARNING"
            elif isinstance(result, bool):
                str_result = result
                if not result:
                    level = "WARNING"
            elif hasattr(result, "response") and isinstance(result.response, list) and len(result.response) > 0:
                str_result = result.response[0]
                if isinstance(str_result, dict):
                    result_dict = json.loads(str_result)
                    result_state = result_dict.get("result")
                    if result_state == 1:
                        level = "WARNING"
            else:
                str_result = None

            insert_logging_api(owner, level, class_nm,
                               payload, ip_address, str_result)
            return result
        except Exception as e:
            level = "WARNING"
            insert_logging_api(owner, level, class_nm,
                               payload, ip_address, None)
            raise e
    return wrapped


def insert_logging_api(owner, log_level, funcation, payload, ip_address, result):
    if payload and payload.get('passwd'):
        payload['passwd'] = hashlib.sha256(
            payload['passwd'].encode('utf-8')).hexdigest()

    message = {
        'user_id': owner,
        'ip_address': ip_address,
        'function': funcation,
        'log_level': log_level,
        'payload': json.dumps(payload)
    }
    if log_level == "WARNING":
        # if result:
        #     logging.warning({'result': result})
        logging.warning(message)
    else:
        # if result:
        #     logging.info({'result': result})
        logging.info(message)


def insert_logging_account(owner, action, message):
    ip_address = request.remote_addr
    message = {
        'ip address': ip_address,
        'log level': 'info',
        'action': action,
        'message': message
    }
    logging.info(message)
