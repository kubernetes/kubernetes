// this file handles outputting usage instructions,
// failures, etc. keeps logging in one place.
var decamelize = require('decamelize'),
  wordwrap = require('wordwrap'),
  wsize = require('window-size');

module.exports = function (yargs) {
    var self = {};

    // methods for ouputting/building failure message.
    var fails = [];
    self.failFn = function (f) {
        fails.push(f);
    };

    var failMessage = null;
    var showHelpOnFail = true;
    self.showHelpOnFail = function (enabled, message) {
        if (typeof enabled === 'string') {
            message = enabled;
            enabled = true;
        }
        else if (typeof enabled === 'undefined') {
            enabled = true;
        }
        failMessage = message;
        showHelpOnFail = enabled;
        return self;
    };

    self.fail = function (msg) {
        if (fails.length) {
            fails.forEach(function (f) {
                f(msg);
            });
        } else {
            if (showHelpOnFail) yargs.showHelp("error");
            if (msg) console.error(msg);
            if (failMessage) {
                if (msg) console.error("");
                console.error(failMessage);
            }
            if (yargs.getExitProcess()){
                process.exit(1);
            }else{
                throw new Error(msg);
            }
        }
    };

    // methods for ouputting/building help (usage) message.
    var usage;
    self.usage = function (msg) {
        usage = msg;
    };

    var examples = [];
    self.example = function (cmd, description) {
        examples.push([cmd, description || '']);
    };

    var commands = [];
    self.command = function (cmd, description) {
        commands.push([cmd, description || '']);
    };
    self.getCommands = function () {
        return commands;
    };

    var descriptions = {};
    self.describe = function (key, desc) {
        if (typeof key === 'object') {
            Object.keys(key).forEach(function (k) {
                self.describe(k, key[k]);
            });
        }
        else {
            descriptions[key] = desc;
        }
    };
    self.getDescriptions = function() {
        return descriptions;
    }

    var epilog;
    self.epilog = function (msg) {
        epilog = msg;
    };

    var wrap = windowWidth();
    self.wrap = function (cols) {
        wrap = cols;
    };

    self.help = function () {
        var demanded = yargs.getDemanded(),
            options = yargs.getOptions(),
            keys = Object.keys(
                Object.keys(descriptions)
                .concat(Object.keys(demanded))
                .concat(Object.keys(options.default))
                .reduce(function (acc, key) {
                    if (key !== '_') acc[key] = true;
                    return acc;
                }, {})
            );

        var help = keys.length ? [ 'Options:' ] : [];

        // your application's commands, i.e., non-option
        // arguments populated in '_'.
        if (commands.length) {
            help.unshift('');

            var commandsTable = {};
            commands.forEach(function(command) {
                commandsTable[command[0]] = {
                    desc: command[1],
                    extra: ''
                };
            });

            help = ['Commands:'].concat(formatTable(commandsTable, 5), help);
        }

        // the usage string.
        if (usage) {
            var u = usage.replace(/\$0/g, yargs.$0);
            if (wrap) u = wordwrap(0, wrap)(u);
            help.unshift(u, '');
        }

        // the options table.
        var aliasKeys = (Object.keys(options.alias) || [])
            .concat(Object.keys(yargs.parsed.newAliases) || []);

        keys = keys.filter(function(key) {
            return !yargs.parsed.newAliases[key] && aliasKeys.every(function(alias) {
                return -1 == (options.alias[alias] || []).indexOf(key);
            });
        });

        var switches = keys.reduce(function (acc, key) {
            acc[key] = [ key ].concat(options.alias[key] || [])
                .map(function (sw) {
                    return (sw.length > 1 ? '--' : '-') + sw
                })
                .join(', ')
            ;
            return acc;
        }, {});

        var switchTable = {};
        keys.forEach(function (key) {
            var kswitch = switches[key];
            var desc = descriptions[key] || '';
            var type = null;

            if (options.boolean[key]) type = '[boolean]';
            if (options.count[key]) type = '[count]';
            if (options.string[key]) type = '[string]';
            if (options.normalize[key]) type = '[string]';

            var extra = [
                type,
                demanded[key]
                    ? '[required]'
                    : null
                ,
                defaultString(options.default[key], options.defaultDescription[key])
            ].filter(Boolean).join('  ');

            switchTable[kswitch] = {
              desc: desc,
              extra: extra
            };
        });
        help.push.apply(help, formatTable(switchTable, 3));

        if (keys.length) help.push('');

        // describe some common use-cases for your application.
        if (examples.length) {
            examples.forEach(function (example) {
                example[0] = example[0].replace(/\$0/g, yargs.$0);
            });

            var examplesTable = {};
            examples.forEach(function(example) {
                examplesTable[example[0]] = {
                    desc: example[1],
                    extra: ''
                };
            });

            help.push.apply(help, ['Examples:'].concat(formatTable(examplesTable, 5), ''));
        }

        // the usage string.
        if (epilog) {
            var e = epilog.replace(/\$0/g, yargs.$0);
            if (wrap) e = wordwrap(0, wrap)(e);
            help.push(e, '');
        }

        return help.join('\n');
    };

    self.showHelp = function (level) {
        level = level || 'error';
        console[level](self.help());
    }

    // format the default-value-string displayed in
    // the right-hand column.
    function defaultString(value, defaultDescription) {
        var string = '[default: ';

        if (value === undefined) return null;

        if (defaultDescription) {
          string += defaultDescription;
        } else {
          switch (typeof value) {
              case 'string':
                string += JSON.stringify(value);
                break;
              case 'function':
                string += '(' + (value.name.length ? decamelize(value.name, '-') : 'generated-value') + ')'
                break;
              default:
                string += value;
          }
        }

        return string + ']';
    }

    // word-wrapped two-column layout used by
    // examples, options, commands.
    function formatTable (table, padding) {
        var output = [];

        // size of left-hand-column.
        var llen = longest(Object.keys(table));

        // don't allow the left-column to take up
        // more than half of the screen.
        if (wrap) {
            llen = Math.min(llen, parseInt(wrap / 2));
        }

        // size of right-column.
        var desclen = longest(Object.keys(table).map(function (k) {
            return table[k].desc;
        }));

        Object.keys(table).forEach(function(left) {
            var desc = table[left].desc,
              extra = table[left].extra,
              leftLines = null;

            if (wrap) {
                desc = wordwrap(llen + padding + 1, wrap)(desc)
                    .slice(llen + padding + 1);
            }

            // if we need to wrap the left-hand-column,
            // split it on to multiple lines.
            if (wrap && left.length > llen) {
                leftLines = wordwrap(2, llen)(left.trim()).split('\n');
                left = '';
            }

            var lpadding = new Array(
                Math.max(llen - left.length + padding, 0)
            ).join(' ');

            var dpadding = new Array(
                Math.max(desclen - desc.length + 1, 0)
            ).join(' ');

            if (!wrap && dpadding.length > 0) {
                desc += dpadding;
            }

            var prelude = '  ' + left + lpadding;

            var body = [ desc, extra ].filter(Boolean).join('  ');

            if (wrap) {
                var dlines = desc.split('\n');
                var dlen = dlines.slice(-1)[0].length
                    + (dlines.length === 1 ? prelude.length : 0)

                if (extra.length > wrap) {
                    body = desc + '\n' + wordwrap(llen + 4, wrap)(extra)
                } else {
                    body = desc + (dlen + extra.length > wrap - 2
                        ? '\n'
                            + new Array(wrap - extra.length + 1).join(' ')
                            + extra
                        : new Array(wrap - extra.length - dlen + 1).join(' ')
                            + extra
                    );
                }
            }

            if (leftLines) { // handle word-wrapping the left-hand-column.
              var rightLines = body.split('\n'),
                firstLine = prelude + rightLines[0],
                lineCount = Math.max(leftLines.length, rightLines.length);

              for (var i = 0; i < lineCount; i++) {
                var left = leftLines[i],
                  right = i ? rightLines[i] : firstLine;

                output.push(strcpy(left, right, firstLine.length));
              }
            } else {
              output.push(prelude + body);
            }
        });

        return output;
    }

    // find longest string in array of strings.
    function longest (xs) {
        return Math.max.apply(
            null,
            xs.map(function (x) { return x.length })
        );
    }

    // copy one string into another, used when
    // formatting usage table.
    function strcpy (source, destination, width) {
        var str = ''

        source = source || '';
        destination = destination || new Array(width).join(' ');

        for (var i = 0; i < destination.length; i++) {
          var char = destination.charAt(i);

          if (char === ' ') char = source.charAt(i) || char;

          str += char;
        }

        return str;
    }

    // guess the width of the console window, max-width 100.
    function windowWidth() {
        return wsize.width ? Math.min(80, wsize.width) : null;
    }

    // logic for displaying application version.
    var version = null;
    self.version = function (ver, opt, msg) {
        version = ver;
    };

    self.showVersion = function() {
        if (typeof version === 'function') console.log(version());
        else console.log(version);
    };

    return self;
}
