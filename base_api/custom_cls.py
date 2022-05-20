# -*- coding: UTF-8 -*-
import json
import re

import six
from werkzeug.wrappers import BaseResponse

from flask import current_app, redirect, request
from flask._compat import with_metaclass
from flask.views import MethodViewType, View
from flask_restplus import Api as BaseApi
from flask_restplus import Namespace, Resource
from flask_restplus.errors import abort as errors_abort
from flask_restplus.model import Draft4Validator, Model, ValidationError
from flask_restplus.reqparse import (Argument, HTTPStatus, ParseResult,
                                     RequestParser)
from flask_restplus.utils import unpack
from utils.log_utils import *
import logging

RE_REQUIRED = re.compile(r'u?\'(?P<name>.*)\' is a required property', re.I | re.U)
RE_TYPE = re.compile(r'.* is not of type u?\'(?P<type>.*)\'', re.I | re.U)
RE_ENUM = re.compile(r'.* is not one of (?P<enum>.*)', re.I | re.U)

SPLIT_CHAR = ','
_friendly_location = {
    'json': 'JSON Body',
    'form': 'Post Body',
    'args': 'Query String',
    'values': 'Post Body 或 Query String',
    'headers': 'HTTP Headers',
    'cookies': 'Request\'s Cookies',
    'files': 'Uploaded File',
}


class CustomArgument(Argument):
    def handle_validation_error(self, error, bundle_errors):
        '''
        Called when an error is raised while parsing. Aborts the request
        with a 400 status and an error message

        :param error: the error that was raised
        :param bool bundle_errors: do not abort when first error occurs, return a
            dict with the name of the argument and the error message to be
            bundled
        '''
        errors = {self.name: error}

        if bundle_errors:
            return ValueError(error), errors
        errors_abort(HTTPStatus.BAD_REQUEST, message=json.dumps(errors, ensure_ascii=False), result=1)

    def parse(self, request, bundle_errors=False):
        '''
        Parses argument value(s) from the request, converting according to
        the argument's type.

        :param request: The flask request object to parse arguments from
        :param bool bundle_errors: do not abort when first error occurs, return a
            dict with the name of the argument and the error message to be
            bundled
        '''
        bundle_errors = current_app.config.get('BUNDLE_ERRORS', False) or bundle_errors
        source = self.source(request)

        results = []

        # Sentinels
        _not_found = False
        _found = True

        for operator in self.operators:
            name = self.name + operator.replace('=', '', 1)
            if name in source:
                # Account for MultiDict and regular dict
                if hasattr(source, 'getlist'):
                    values = source.getlist(name)
                else:
                    values = [source.get(name)]

                for value in values:
                    if hasattr(value, 'strip') and self.trim:
                        value = value.strip()
                    if hasattr(value, 'lower') and not self.case_sensitive:
                        value = value.lower()

                        if hasattr(self.choices, '__iter__'):
                            self.choices = [choice.lower() for choice in self.choices]

                    try:
                        if self.action == 'split':
                            value = [self.convert(v, operator) for v in value.split(SPLIT_CHAR)]
                        else:
                            value = self.convert(value, operator)
                    except Exception as error:
                        if self.ignore:
                            continue
                        return self.handle_validation_error(error, bundle_errors)

                    if self.choices and value not in self.choices:
                        msg = 'The value \'{0}\' is not a valid choice for \'{1}\'.'.format(value, name)
                        return self.handle_validation_error(msg, bundle_errors)

                    if name in request.unparsed_arguments:
                        request.unparsed_arguments.pop(name)
                    results.append(value)

        if not results and self.required:
            if isinstance(self.location, six.string_types):
                location = _friendly_location.get(self.location, self.location)
            else:
                locations = [_friendly_location.get(loc, loc) for loc in self.location]
                location = ' or '.join(locations)
            error_msg = '在{0}為缺少必要的參數'.format(location)
            return self.handle_validation_error(error_msg, bundle_errors)

        if not results:
            if callable(self.default):
                return self.default(), _not_found
            else:
                return self.default, _not_found

        if self.action == 'append':
            return results, _found

        if self.action == 'store' or len(results) == 1:
            return results[0], _found
        return results, _found


class CustomModel(Model):
    def validate(self, data, resolver=None, format_checker=None):
        validator = Draft4Validator(self.__schema__, resolver=resolver, format_checker=format_checker)
        try:
            validator.validate(data)
        except ValidationError:
            errors = [self.format_error(e) for e in validator.iter_errors(data)]
            errors_abort(HTTPStatus.BAD_REQUEST, message=json.dumps(errors, ensure_ascii=False), result=1)

    def format_error(self, error):
        path = list(error.path)
        if error.validator == 'required':
            name = RE_REQUIRED.match(error.message).group('name')
            path.append(name)
            key = '.'.join(str(p) for p in path)
            message = "payload中參數'%s'為必填的參數" % key
        elif error.validator == 'type':
            t = RE_TYPE.match(error.message).group("type")
            key = '.'.join(str(p) for p in path)
            message = "payload中參數'%s'只能接受'%s'的類型" % (key, t)
        elif error.validator == 'enum':
            key = '.'.join(str(p) for p in path)
            enum = RE_ENUM.match(error.message).group("enum")
            message = "payload中參數'%s'只能接受數值為%s" % (key, enum)
        else:
            message = ""
        return message


class CustomRequestParser(RequestParser):
    def __init__(self, argument_class=CustomArgument, result_class=ParseResult,
                 trim=False, bundle_errors=False):
        super(CustomRequestParser, self).__init__(argument_class, result_class, trim, bundle_errors)


class CustomNamespace(Namespace):
    def model(self, name=None, model=None, mask=None, **kwargs):
        '''
        Register a model

        .. seealso:: :class:`Model`
        '''
        models = CustomModel(name, model, mask=mask)
        models.__apidoc__.update(kwargs)
        return self.add_model(name, models)

    def extend(self, name, parent, fields):
        '''
        Extend a model (Duplicate all fields)

        :deprecated: since 0.9. Use :meth:`clone` instead
        '''
        if isinstance(parent, list):
            parents = parent + [fields]
            model = CustomModel.extend(name, *parents)
        else:
            model = CustomModel.extend(name, parent, fields)
        return self.add_model(name, model)

    def clone(self, name, *specs):
        '''
        Clone a model (Duplicate all fields)

        :param str name: the resulting model name
        :param specs: a list of models from which to clone the fields

        .. seealso:: :meth:`Model.clone`

        '''
        model = CustomModel.clone(name, *specs)
        return self.add_model(name, model)

    def inherit(self, name, *specs):
        '''
        Inherit a modal (use the Swagger composition pattern aka. allOf)

        .. seealso:: :meth:`Model.inherit`
        '''
        model = CustomModel.inherit(name, *specs)
        return self.add_model(name, model)

    def parser(self):
        '''Instanciate a :class:`~RequestParser`'''
        return CustomRequestParser()

    def expect(self, *inputs, **kwargs):
        '''
        A decorator to Specify the expected input model

        :param ModelBase|Parse inputs: An expect model or request parser
        :param bool validate: whether to perform validation or not

        '''
        expect = []
        params = {
            'validate': kwargs.get('validate', None) or self._validate,
            'expect': expect
        }
        for param in inputs:
            expect.append(param)
        return self.doc(**params)


class CustomMethodView(with_metaclass(MethodViewType, View)):
    allow_roles = []

    def dispatch_request(self, *args, **kwargs):
        meth = getattr(self, request.method.lower(), None)

        if meth is None:
            # 如果該request meth 不存在，則轉至首頁(/)
            return redirect("/", code=302)

        if self.allow_roles:
            allow_roles = self.allow_roles if isinstance(self.allow_roles, list) else [self.allow_roles]
            # 無權限，回首頁(/)
            # if not check_roles_permission(*allow_roles):
            #     return redirect("/", code=302)
        return meth(*args, **kwargs)


class CustomResource(Resource, CustomMethodView):
    allow_roles = []
    check_permission = True
    def __init__(self, *args, **kwargs):
        super(CustomResource, self).__init__(*args, **kwargs)

    @auto_write_logging_api
    def dispatch_request(self, *args, **kwargs):
        # Taken from flask
        #noinspection PyUnresolvedReferences
        meth = getattr(self, request.method.lower(), None)
        if meth is None and request.method == 'HEAD':
            meth = getattr(self, 'get', None)
        assert meth is not None, 'Unimplemented method %r' % request.method

        self.validate_payload(meth)

        # if self.check_permission:
        #     check_permission()

        if self.allow_roles:
            allow_roles = self.allow_roles if isinstance(self.allow_roles, list) else [self.allow_roles]
        
        from datetime import datetime, timedelta
        from flask import session
        session['refresh_token_expire_time'] = (datetime.now() + current_app.config["JWT_REFRESH_TOKEN_EXPIRES"]).strftime("%Y-%m-%d %H:%M:%S")

        resp = meth(*args, **kwargs)

        if isinstance(resp, BaseResponse):
            return resp
        representations = self.representations or {}

        mediatype = request.accept_mimetypes.best_match(representations, default=None)
        if mediatype in representations:
            data, code, headers = unpack(resp)
            resp = representations[mediatype](data, code, headers)
            resp.headers['Content-Type'] = mediatype
            return resp
        return resp


class Api(BaseApi):
    def namespace(self, *args, **kwargs):
        '''
        A namespace factory.

        :returns Namespace: a new namespace instance
        '''
        kwargs['ordered'] = kwargs.get('ordered', self.ordered)
        ns = CustomNamespace(*args, **kwargs)
        self.add_namespace(ns)
        return ns
