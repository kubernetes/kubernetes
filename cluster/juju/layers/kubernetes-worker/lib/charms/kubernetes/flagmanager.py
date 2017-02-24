#!/usr/bin/env python

# Copyright 2015 The Kubernetes Authors.
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

from charmhelpers.core import unitdata


class FlagManager:
    '''
    FlagManager - A Python class for managing the flags to pass to an
    application without remembering what's been set previously.

    This is a blind class assuming the operator knows what they are doing.
    Each instance of this class should be initialized with the intended
    application to manage flags. Flags are then appended to a data-structure
    and cached in unitdata for later recall.

    THe underlying data-provider is backed by a SQLITE database on each unit,
    tracking the dictionary, provided from the 'charmhelpers' python package.
    Summary:
    opts = FlagManager('docker')
    opts.add('bip', '192.168.22.2')
    opts.to_s()
    '''

    def __init__(self, daemon, opts_path=None):
        self.db = unitdata.kv()
        self.daemon = daemon
        if not self.db.get(daemon):
            self.data = {}
        else:
            self.data = self.db.get(daemon)

    def __save(self):
        self.db.set(self.daemon, self.data)

    def add(self, key, value, strict=False):
        '''
        Adds data to the map of values for the DockerOpts file.
        Supports single values, or "multiopt variables". If you
        have a flag only option, like --tlsverify, set the value
        to None. To preserve the exact value, pass strict
        eg:
        opts.add('label', 'foo')
        opts.add('label', 'foo, bar, baz')
        opts.add('flagonly', None)
        opts.add('cluster-store', 'consul://a:4001,b:4001,c:4001/swarm',
                 strict=True)
        '''
        if strict:
            self.data['{}-strict'.format(key)] = value
            self.__save()
            return

        if value:
            values = [x.strip() for x in value.split(',')]
            # handle updates
            if key in self.data and self.data[key] is not None:
                item_data = self.data[key]
                for c in values:
                    c = c.strip()
                    if c not in item_data:
                        item_data.append(c)
                self.data[key] = item_data
            else:
                # handle new
                self.data[key] = values
        else:
            # handle flagonly
            self.data[key] = None
        self.__save()

    def remove(self, key, value):
        '''
        Remove a flag value from the DockerOpts manager
        Assuming the data is currently {'foo': ['bar', 'baz']}
        d.remove('foo', 'bar')
        > {'foo': ['baz']}
        :params key:
        :params value:
        '''
        self.data[key].remove(value)
        self.__save()

    def destroy(self, key, strict=False):
        '''
        Destructively remove all values and key from the FlagManager
        Assuming the data is currently {'foo': ['bar', 'baz']}
        d.wipe('foo')
        >{}
        :params key:
        :params strict:
        '''
        try:
            if strict:
                self.data.pop('{}-strict'.format(key))
            else:
                self.data.pop('key')
        except KeyError:
            pass

    def to_s(self):
        '''
        Render the flags to a single string, prepared for the Docker
        Defaults file. Typically in /etc/default/docker
        d.to_s()
        > "--foo=bar --foo=baz"
        '''
        flags = []
        for key in self.data:
            if self.data[key] is None:
                # handle flagonly
                flags.append("{}".format(key))
            elif '-strict' in key:
                # handle strict values, and do it in 2 steps.
                # If we rstrip -strict it strips a tailing s
                proper_key = key.rstrip('strict').rstrip('-')
                flags.append("{}={}".format(proper_key, self.data[key]))
            else:
                # handle multiopt and typical flags
                for item in self.data[key]:
                    flags.append("{}={}".format(key, item))
        return ' '.join(flags)
