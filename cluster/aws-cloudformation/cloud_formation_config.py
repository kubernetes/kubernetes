#!/usr/bin/env python

"""CLI for interacting with AWS resources via jsonnet cloudformation configs."""

# Copyright 2016 The Kubernetes Authors All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import datetime
import difflib
import json
import logging
import socket
import os
import pprint
import sys
import uuid
from time import sleep

import _jsonnet as jsonnet
import boto3
from termcolor import cprint, colored

STATUS_POLLING_DELAY_SECONDS = 1


class StackCache(object):
    """A cache of CloudFormation stack descriptions.

    Used to pull dependent parameters from existing stacks to feed into the current stack.
    """
    def __init__(self):
        self.client = boto3.client('cloudformation')
        self.cache = {}

    def get(self, stack_name):
        if stack_name not in self.cache:
            response = self.client.describe_stacks(StackName=stack_name)
            if len(response['Stacks']) != 1:
                raise ValueError('Unable to locate stack %s' % stack_name)
            stack = response['Stacks'][0]
            self.cache[stack_name] = stack

        return self.cache[stack_name]


class CloudFormationConfig(object):
    """A materialized CloudFormation configuration """
    def __init__(self, args):
        self.args = args
        self.parsed_config = None
        self.dependent_params = {}

        self.load_config()
        self.resolve_stack_parameters()

    def load_config(self):
        """Load the jsonnet config file from disk and parse it."""
        resolved_file_path = os.path.expandvars(os.path.expanduser(self.args.configuration))
        logging.debug('Resolved file to %s', resolved_file_path)

        external_vars = {
            'USER': os.environ['USER'],
            'HOSTNAME': socket.gethostname(),
            'TIMESTAMP': str(datetime.datetime.now())
        }
        json_blob = jsonnet.evaluate_file(resolved_file_path, ext_vars=external_vars)
        logging.debug('Jsonnet-to-json: %s', pprint.pformat(json_blob, indent=2))

        self.parsed_config = json.loads(json_blob)
        logging.debug('Parsed to: %s', pprint.pformat(self.parsed_config, indent=2))

        logging.info('Operating on stack: %s', self.parsed_config['Metadata']['StackName'])

    def resolve_stack_parameters(self):
        """Pull dependent stack output parameters from AWS to feed into this config."""
        stack_cache = StackCache()

        for param_source in self.param_sources:
            if isinstance(param_source, list):
                raise ValueError('Invalid ParamSource %s, expected object but received array.' % param_source)

            parameter_name = param_source['Param']
            source_stack_name = param_source['Source']['Stack']
            source_output_name = param_source['Source']['Output']

            source_stack = stack_cache.get(source_stack_name)
            parameter_value = None

            for param in source_stack['Outputs']:
                if param['OutputKey'] == source_output_name:
                    parameter_value = param['OutputValue']
                    break

            if not parameter_value:
                raise ValueError('Unable to locate output parameter %s in stack %s' % (
                    source_output_name, source_stack_name))

            self.dependent_params[parameter_name] = parameter_value

    @property
    def param_sources(self):
        return self.parsed_config['Metadata']['ParamSources']

    @property
    def stack_name(self):
        return self.parsed_config['Metadata']['StackName']

    @property
    def region(self):
        return self.parsed_config['Metadata']['Region']

    @property
    def required_caps(self):
        return self.parsed_config['Metadata']['RequiredCaps']

    @property
    def resolved_parameters(self):
        return self.dependent_params

    @property
    def body(self):
        return self.parsed_config


class Action(object):
    """Base class for command actions."""
    def __init__(self):
        self.client = boto3.client('cloudformation')
        self.pp = pprint.PrettyPrinter(indent=2)

    @staticmethod
    def get_config_param_array(config):
        stack_parameters = []
        for key, value in config.resolved_parameters.iteritems():
            stack_parameters.append({'ParameterKey': key, 'ParameterValue': value})
        return stack_parameters

    @staticmethod
    def get_action(command):
        if command == 'print':
            return Print()
        elif command == 'diff':
            return Diff()
        elif command == 'get':
            return Get()
        elif command == 'status':
            return Status()
        elif command == 'create':
            return Create()
        elif command == 'delete':
            return Delete()
        elif command == 'update':
            return Update()


class Print(Action):
    def execute(self, config):
        print('Printing local stack: %s in region %s', config.stack_name, config.region)
        print('-----------------------------------')
        print('Input parameters:')
        print('\n' + self.pp.pformat(config.dependent_params))
        print('-----------------------------------')
        print('Config:')
        print('\n' + self.pp.pformat(config.body))


class Diff(Action):
    def execute(self, config):
        response = self.client.get_template(StackName=config.stack_name)
        current_template_body = response['TemplateBody']

        print('Local Diff')
        print('------------------------')
        from_text = self.pp.pformat(current_template_body).splitlines(1)
        to_text = self.pp.pformat(config.body).splitlines(1)
        for line in difflib.unified_diff(from_text, to_text):
            if line.startswith('-'):
                line = colored(line, 'red')
            elif line.startswith('+'):
                line = colored(line, 'green')
            sys.stdout.write(line)

        print('AWS Calculated Actions')
        print('------------------------')
        change_set_name = 'diff-%s' % str(uuid.uuid4())
        response = self.client.create_change_set(StackName=config.stack_name, Capabilities=config.required_caps,
                                                 TemplateBody=json.dumps(config.body), UsePreviousTemplate=False,
                                                 Parameters=self.get_config_param_array(config),
                                                 ChangeSetName=change_set_name, Description='diff')
        change_set_arn = response['Id']

        try:
            terminal_states = ['CREATE_COMPLETE', 'DELETE_COMPLETE', 'FAILED']
            while True:
                response = self.client.describe_change_set(ChangeSetName=change_set_arn)
                if response['Status'] in terminal_states:
                    break
                sleep(STATUS_POLLING_DELAY_SECONDS)

            changes = response['Changes']
            for change in changes:
                self.pp.pprint(change['ResourceChange'])
        finally:
            self.client.delete_change_set(ChangeSetName=change_set_arn)


class Get(Action):
    def execute(self, config):
        response = self.client.get_template(StackName=config.stack_name)
        current_template_body = response['TemplateBody']
        print('Printing live stack: %s in region %s', config.stack_name, config.region)
        print('-----------------------------------')
        print('\n' + self.pp.pformat(current_template_body))


class Status(Action):
    def execute(self, config):
        last_status_timestamp = None

        while True:
            try:
                status = self.client.describe_stack_events(StackName=config.stack_name)
            except:
                return
            self.report_status(status, last_status_timestamp, True)

            # Return if we're not watching for updates or the operation has completed
            first_event = status['StackEvents'][0]
            timestamp = first_event['Timestamp']
            newer = last_status_timestamp and timestamp > last_status_timestamp
            if not config.args.watch or (newer and self.is_update_complete(first_event)):
                break

            last_status_timestamp = timestamp
            sleep(STATUS_POLLING_DELAY_SECONDS)

    def report_status(self, status, filter_timestamp=None, reverse_order=False):
        if not filter_timestamp:
            print('[timestamp, stackName, resourceType, logicalId, physicalId, status, statusReason]')

        events = status['StackEvents']

        for index, event in enumerate(events):
            if reverse_order:
                event = events[len(events) - index - 1]
            if (not filter_timestamp) or (filter_timestamp and event['Timestamp'] > filter_timestamp):
                resource_status = event['ResourceStatusReason'] if 'ResourceStatusReason' in event else ''
                self.colorized_print('%s, %s, %s, %s, %s, %s, %s' % (
                    event['Timestamp'], event['StackName'], event['ResourceStatus'], event['ResourceType'],
                    event['LogicalResourceId'], event['PhysicalResourceId'], resource_status))

    @staticmethod
    def colorized_print(line):
        if '_ROLLBACK_' in line:
            cprint(line, 'red')
        elif '_FAILED' in line:
            cprint(line, 'yellow')
        elif '_COMPLETE, AWS::CloudFormation::Stack' in line:
            cprint(line, 'green')
        elif '_IN_PROGRESS, AWS::CloudFormation::Stack' in line:
            cprint(line, 'cyan')
        else:
            print line

    @staticmethod
    def is_update_complete(event):
        completion_events = [
            'CREATE_FAILED', 'CREATE_COMPLETE', 'CREATE_ROLLBACK_COMPLETE',
            'DELETE_FAILED', 'DELETE_COMPLETE', 'DELETE_SKIPPED', 'DELETE_ROLLBACK_COMPLETE',
            'UPDATE_FAILED', 'UPDATE_COMPLETE', 'UPDATE_ROLLBACK_COMPLETE'
        ]
        return event['ResourceType'] == 'AWS::CloudFormation::Stack' and event['ResourceStatus'] in completion_events


class Create(Status):
    def execute(self, config):
        self.client.create_stack(StackName=config.stack_name, Capabilities=config.required_caps,
                                 TemplateBody=json.dumps(config.body), Parameters=self.get_config_param_array(config))

        return super(Create, self).execute(config)


class Update(Status):
    def execute(self, config):
        self.client.update_stack(StackName=config.stack_name, Capabilities=config.required_caps,
                                 TemplateBody=json.dumps(config.body), Parameters=self.get_config_param_array(config))

        return super(Update, self).execute(config)


class Delete(Status):
    def execute(self, config):
        self.client.delete_stack(StackName=config.stack_name)
        return super(Delete, self).execute(config)


def parse_args():
    """Defines and parses command line arguments for this tool."""
    parser = argparse.ArgumentParser(
        description='Interact with AWS using jsonnet-based CloudFormation declarative configuration files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('configuration', help='Jsonnet file to act upon')
    parser.add_argument('command', help='Action to take',
                        choices=['create', 'delete', 'update', 'print', 'diff', 'status', 'get'])
    parser.add_argument('--verbose', help='Increased logging for debugging', action='store_true')
    parser.add_argument('--no-watch', help='Return immediately instead of tailing the activity log.',
                        action='store_false', dest='watch')
    parser.set_defaults(watch=True)

    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=log_level)

    config = CloudFormationConfig(args)
    command = Action.get_action(args.command)
    return command.execute(config)


if __name__ == "__main__":
    main()
