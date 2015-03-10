#!/usr/bin/python

import yaml, codecs, sys, os.path, optparse

class Style:

    def __init__(self, header=None, footer=None,
            tokens=None, events=None, replaces=None):
        self.header = header
        self.footer = footer
        self.replaces = replaces
        self.substitutions = {}
        for domain, Class in [(tokens, 'Token'), (events, 'Event')]:
            if not domain:
                continue
            for key in domain:
                name = ''.join([part.capitalize() for part in key.split('-')])
                cls = getattr(yaml, '%s%s' % (name, Class))
                value = domain[key]
                if not value:
                    continue
                start = value.get('start')
                end = value.get('end')
                if start:
                    self.substitutions[cls, -1] = start
                if end:
                    self.substitutions[cls, +1] = end

    def __setstate__(self, state):
        self.__init__(**state)

yaml.add_path_resolver(u'tag:yaml.org,2002:python/object:__main__.Style',
        [None], dict)
yaml.add_path_resolver(u'tag:yaml.org,2002:pairs',
        [None, u'replaces'], list)

class YAMLHighlight:

    def __init__(self, options):
        config = yaml.load(file(options.config, 'rb').read())
        self.style = config[options.style]
        if options.input:
            self.input = file(options.input, 'rb')
        else:
            self.input = sys.stdin
        if options.output:
            self.output = file(options.output, 'wb')
        else:
            self.output = sys.stdout

    def highlight(self):
        input = self.input.read()
        if input.startswith(codecs.BOM_UTF16_LE):
            input = unicode(input, 'utf-16-le')
        elif input.startswith(codecs.BOM_UTF16_BE):
            input = unicode(input, 'utf-16-be')
        else:
            input = unicode(input, 'utf-8')
        substitutions = self.style.substitutions
        tokens = yaml.scan(input)
        events = yaml.parse(input)
        markers = []
        number = 0
        for token in tokens:
            number += 1
            if token.start_mark.index != token.end_mark.index:
                cls = token.__class__
                if (cls, -1) in substitutions:
                    markers.append([token.start_mark.index, +2, number, substitutions[cls, -1]])
                if (cls, +1) in substitutions:
                    markers.append([token.end_mark.index, -2, number, substitutions[cls, +1]])
        number = 0
        for event in events:
            number += 1
            cls = event.__class__
            if (cls, -1) in substitutions:
                markers.append([event.start_mark.index, +1, number, substitutions[cls, -1]])
            if (cls, +1) in substitutions:
                markers.append([event.end_mark.index, -1, number, substitutions[cls, +1]])
        markers.sort()
        markers.reverse()
        chunks = []
        position = len(input)
        for index, weight1, weight2, substitution in markers:
            if index < position:
                chunk = input[index:position]
                for substring, replacement in self.style.replaces:
                    chunk = chunk.replace(substring, replacement)
                chunks.append(chunk)
                position = index
            chunks.append(substitution)
        chunks.reverse()
        result = u''.join(chunks)
        if self.style.header:
            self.output.write(self.style.header)
        self.output.write(result.encode('utf-8'))
        if self.style.footer:
            self.output.write(self.style.footer)

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-s', '--style', dest='style', default='ascii',
            help="specify the highlighting style", metavar='STYLE')
    parser.add_option('-c', '--config', dest='config',
            default=os.path.join(os.path.dirname(sys.argv[0]), 'yaml_hl.cfg'),
            help="set an alternative configuration file", metavar='CONFIG')
    parser.add_option('-i', '--input', dest='input', default=None,
            help="set the input file (default: stdin)", metavar='FILE')
    parser.add_option('-o', '--output', dest='output', default=None,
            help="set the output file (default: stdout)", metavar='FILE')
    (options, args) = parser.parse_args()
    hl = YAMLHighlight(options)
    hl.highlight()

