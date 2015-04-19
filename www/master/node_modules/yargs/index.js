var path = require('path');
var minimist = require('./lib/minimist');
var wordwrap = require('./lib/wordwrap');

/*  Hack an instance of Argv with process.argv into Argv
    so people can do
        require('yargs')(['--beeble=1','-z','zizzle']).argv
    to parse a list of args and
        require('yargs').argv
    to get a parsed version of process.argv.
*/

var inst = Argv(process.argv.slice(2));
Object.keys(inst).forEach(function (key) {
    Argv[key] = typeof inst[key] == 'function'
        ? inst[key].bind(inst)
        : inst[key];
});

var exports = module.exports = Argv;
function Argv (processArgs, cwd) {
    var self = {};
    if (!cwd) cwd = process.cwd();
    
    self.$0 = process.argv
        .slice(0,2)
        .map(function (x) {
            var b = rebase(cwd, x);
            return x.match(/^\//) && b.length < x.length
                ? b : x
        })
        .join(' ')
    ;
    
    if (process.env._ != undefined && process.argv[1] == process.env._) {
        self.$0 = process.env._.replace(
            path.dirname(process.execPath) + '/', ''
        );
    }

    var options;
    self.resetOptions = function () {
        options = {
            boolean: [],
            string: [],
            alias: {},
            default: [],
            requiresArg: [],
            count: [],
            normalize: [],
            config: []
        };
        return self;
    };
    self.resetOptions();
    
    self.boolean = function (bools) {
        options.boolean.push.apply(options.boolean, [].concat(bools));
        return self;
    };

    self.normalize = function (strings) {
        options.normalize.push.apply(options.normalize, [].concat(strings));
        return self;
    };

    self.config = function (configs) {
        options.config.push.apply(options.config, [].concat(configs));
        return self;
    };

    var examples = [];
    self.example = function (cmd, description) {
        examples.push([cmd, description]);
        return self;
    };
    
    self.string = function (strings) {
        options.string.push.apply(options.string, [].concat(strings));
        return self;
    };
    
    self.default = function (key, value) {
        if (typeof key === 'object') {
            Object.keys(key).forEach(function (k) {
                self.default(k, key[k]);
            });
        }
        else {
            options.default[key] = value;
        }
        return self;
    };
    
    self.alias = function (x, y) {
        if (typeof x === 'object') {
            Object.keys(x).forEach(function (key) {
                self.alias(key, x[key]);
            });
        }
        else {
            options.alias[x] = (options.alias[x] || []).concat(y);
        }
        return self;
    };

    self.count = function(counts) {
        options.count.push.apply(options.count, [].concat(counts));
        return self;
    };
    
    var demanded = {};
    self.demand = self.required = self.require = function (keys, msg) {
        if (typeof keys == 'number') {
            if (!demanded._) demanded._ = { count: 0, msg: null };
            demanded._.count += keys;
            demanded._.msg = msg;
        }
        else if (Array.isArray(keys)) {
            keys.forEach(function (key) {
                self.demand(key, msg);
            });
        }
        else {
            if (typeof msg === 'string') {
                demanded[keys] = { msg: msg };
            }
            else if (msg === true || typeof msg === 'undefined') {
                demanded[keys] = { msg: null };
            }
        }
        
        return self;
    };

    self.requiresArg = function (requiresArgs) {
        options.requiresArg.push.apply(options.requiresArg, [].concat(requiresArgs));
        return self;
    };

    var implied = {};
    self.implies = function (key, value) {
        if (typeof key === 'object') {
            Object.keys(key).forEach(function (k) {
                self.implies(k, key[k]);
            });
        } else {
            implied[key] = value;
        }
        return self;
    };

    var usage;
    self.usage = function (msg, opts) {
        if (!opts && typeof msg === 'object') {
            opts = msg;
            msg = null;
        }
        
        usage = msg;
        
        if (opts) self.options(opts);
        
        return self;
    };

    var fails = [];
    self.fail = function (f) {
        fails.push(f);
        return self;
    };

    function fail (msg) {
        if (fails.length) {
            fails.forEach(function (f) {
                f(msg);
            });
        } else {
            if (showHelpOnFail) {
                self.showHelp();
            }
            if (msg) console.error(msg);
            if (failMessage) {
                if (msg) {
                    console.error("");
                }
                console.error(failMessage);
            }
            process.exit(1);
        }
    }
    
    var checks = [];
    self.check = function (f) {
        checks.push(f);
        return self;
    };

    self.defaults = self.default;

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
        return self;
    };
    
    self.parse = function (args) {
        return parseArgs(args);
    };
    
    self.option = self.options = function (key, opt) {
        if (typeof key === 'object') {
            Object.keys(key).forEach(function (k) {
                self.options(k, key[k]);
            });
        }
        else {
            if (opt.alias) self.alias(key, opt.alias);

            var demand = opt.demand || opt.required || opt.require;
            if (demand) {
                self.demand(key, demand);
            }

            if (typeof opt.default !== 'undefined') {
                self.default(key, opt.default);
            }
            
            if (opt.boolean || opt.type === 'boolean') {
                self.boolean(key);
                if (opt.alias) self.boolean(opt.alias);
            }
            if (opt.string || opt.type === 'string') {
                self.string(key);
                if (opt.alias) self.string(opt.alias);
            }
            if (opt.count || opt.type === 'count') {
                self.count(key);
            }

            var desc = opt.describe || opt.description || opt.desc;
            if (desc) {
                self.describe(key, desc);
            }

            if (opt.requiresArg) {
                self.requiresArg(key);
            }
        }

        return self;
    };

    var wrap = null;
    self.wrap = function (cols) {
        wrap = cols;
        return self;
    };

    var strict = false;
    self.strict = function () {
        strict = true;
        return self;
    };
    
    self.showHelp = function (fn) {
        if (!fn) fn = console.error.bind(console);
        fn(self.help());
        return self;
    };

    var version = null;
    var versionOpt = null;
    self.version = function (ver, opt, msg) {
        version = ver;
        versionOpt = opt;
        self.describe(opt, msg || 'Show version number');
        return self;
    };

    var helpOpt = null;
    self.addHelpOpt = function (opt, msg) {
        helpOpt = opt;
        self.describe(opt, msg || 'Show help');
        return self;
    };

    var failMessage = null;
    var showHelpOnFail = true;
    self.showHelpOnFail = function (enabled, message) {
        if (typeof enabled === 'string') {
            enabled = true;
            message = enabled;
        }
        else if (typeof enabled === 'undefined') {
            enabled = true;
        }
        failMessage = message;
        showHelpOnFail = enabled;
        return self;
    };


    self.help = function () {
        if (arguments.length > 0) {
            return self.addHelpOpt.apply(self, arguments);
        }

        var keys = Object.keys(
            Object.keys(descriptions)
            .concat(Object.keys(demanded))
            .concat(Object.keys(options.default))
            .reduce(function (acc, key) {
                if (key !== '_') acc[key] = true;
                return acc;
            }, {})
        );
        
        var help = keys.length ? [ 'Options:' ] : [];

        if (examples.length) {
            help.unshift('');
            examples.forEach(function (example) {
                example[0] = example[0].replace(/\$0/g, self.$0);
            });

            var commandlen = longest(examples.map(function (a) {
                return a[0];
            }));

            var exampleLines = examples.map(function(example) {
                var command = example[0];
                var description = example[1];
                command += Array(commandlen + 5 - command.length).join(' ');
                return '  ' + command + description;
            });

            exampleLines.push('');
            help = exampleLines.concat(help);
            help.unshift('Examples:');
        }

        if (usage) {
            help.unshift(usage.replace(/\$0/g, self.$0), '');
        }

        var aliasKeys = (Object.keys(options.alias) || [])
            .concat(Object.keys(self.parsed.newAliases) || []);

        keys = keys.filter(function(key) {
            return !self.parsed.newAliases[key] && aliasKeys.every(function(alias) {
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

        var switchlen = longest(Object.keys(switches).map(function (s) {
            return switches[s] || '';
        }));
        
        var desclen = longest(Object.keys(descriptions).map(function (d) { 
            return descriptions[d] || '';
        }));
        
        keys.forEach(function (key) {
            var kswitch = switches[key];
            var desc = descriptions[key] || '';
            
            if (wrap) {
                desc = wordwrap(switchlen + 4, wrap)(desc)
                    .slice(switchlen + 4)
                ;
            }
            
            var spadding = new Array(
                Math.max(switchlen - kswitch.length + 3, 0)
            ).join(' ');
            
            var dpadding = new Array(
                Math.max(desclen - desc.length + 1, 0)
            ).join(' ');
            
            var type = null;
            
            if (options.boolean[key]) type = '[boolean]';
            if (options.count[key]) type = '[count]';
            if (options.string[key]) type = '[string]';
            if (options.normalize[key]) type = '[string]';
            
            if (!wrap && dpadding.length > 0) {
                desc += dpadding;
            }
            
            var prelude = '  ' + kswitch + spadding;
            var extra = [
                type,
                demanded[key]
                    ? '[required]'
                    : null
                ,
                options.default[key] !== undefined
                    ? '[default: ' + (typeof options.default[key] === 'string' ?
                    JSON.stringify : String)(options.default[key]) + ']'
                    : null
            ].filter(Boolean).join('  ');
            
            var body = [ desc, extra ].filter(Boolean).join('  ');
            
            if (wrap) {
                var dlines = desc.split('\n');
                var dlen = dlines.slice(-1)[0].length
                    + (dlines.length === 1 ? prelude.length : 0)
                
                body = desc + (dlen + extra.length > wrap - 2
                    ? '\n'
                        + new Array(wrap - extra.length + 1).join(' ')
                        + extra
                    : new Array(wrap - extra.length - dlen + 1).join(' ')
                        + extra
                );
            }
            
            help.push(prelude + body);
        });
        
        if (keys.length) help.push('');
        return help.join('\n');
    };
    
    Object.defineProperty(self, 'argv', {
        get : function () { return parseArgs(processArgs) },
        enumerable : true
    });
    
    function parseArgs (args) {
        var parsed = minimist(args, options),
            argv = parsed.argv,
            aliases = parsed.aliases;

        argv.$0 = self.$0;

        self.parsed = parsed;

        Object.keys(argv).forEach(function(key) {
            if (key === helpOpt) {
                self.showHelp(console.log);
                process.exit(0);
            }
            else if (key === versionOpt) {
                process.stdout.write(version);
                process.exit(0);
            }
        });

        if (demanded._ && argv._.length < demanded._.count) {
            if (demanded._.msg) {
                fail(demanded._.msg);
            } else {
                fail('Not enough non-option arguments: got '
                    + argv._.length + ', need at least ' + demanded._.count
                );
            }
        }

        if (options.requiresArg.length > 0) {
            var missingRequiredArgs = [];

            options.requiresArg.forEach(function(key) {
                var value = argv[key];

                // minimist sets --foo value to true / --no-foo to false
                if (value === true || value === false) {
                    missingRequiredArgs.push(key);
                }
            });

            if (missingRequiredArgs.length == 1) {
                fail("Missing argument value: " + missingRequiredArgs[0]);
            }
            else if (missingRequiredArgs.length > 1) {
                message = "Missing argument values: " + missingRequiredArgs.join(", ");
                fail(message);
            }
        }
        
        var missing = null;
        Object.keys(demanded).forEach(function (key) {
            if (!argv.hasOwnProperty(key)) {
                missing = missing || {};
                missing[key] = demanded[key];
            }
        });
        
        if (missing) {
            var customMsgs = [];
            Object.keys(missing).forEach(function(key) {
                var msg = missing[key].msg;
                if (msg && customMsgs.indexOf(msg) < 0) {
                    customMsgs.push(msg);
                }
            });
            var customMsg = customMsgs.length ? '\n' + customMsgs.join('\n') : '';

            fail('Missing required arguments: ' + Object.keys(missing).join(', ') + customMsg);
        }

        if (strict) {
            var unknown = [];

            var aliases = {};

            Object.keys(parsed.aliases).forEach(function (key) {
                parsed.aliases[key].forEach(function (alias) {
                    aliases[alias] = key;
                });
            });

            Object.keys(argv).forEach(function (key) {
                if (key !== "$0" && key !== "_" &&
                    !descriptions.hasOwnProperty(key) &&
                    !demanded.hasOwnProperty(key) &&
                    !aliases.hasOwnProperty(key)) {
                    unknown.push(key);
                }
            });

            if (unknown.length == 1) {
                fail("Unknown argument: " + unknown[0]);
            }
            else if (unknown.length > 1) {
                fail("Unknown arguments: " + unknown.join(", "));
            }
        }

        checks.forEach(function (f) {
            try {
                var result = f(argv, aliases);
                if (result === false) {
                    fail('Argument check failed: ' + f.toString());
                } else if (typeof result === 'string') {
                    fail(result);
                }
            }
            catch (err) {
                fail(err)
            }
        });

        var implyFail = [];
        Object.keys(implied).forEach(function (key) {
            var num, origKey = key, value = implied[key];

            // convert string '1' to number 1
            var num = Number(key);
            key = isNaN(num) ? key : num;

            if (typeof key === 'number') {
                // check length of argv._
                key = argv._.length >= key;
            } else if (key.match(/^--no-.+/)) {
                // check if key doesn't exist
                key = key.match(/^--no-(.+)/)[1];
                key = !argv[key];
            } else {
                // check if key exists
                key = argv[key];
            }

            num = Number(value);
            value = isNaN(num) ? value : num;

            if (typeof value === 'number') {
                value = argv._.length >= value;
            } else if (value.match(/^--no-.+/)) {
                value = value.match(/^--no-(.+)/)[1];
                value = !argv[value];
            } else {
                value = argv[value];
            }

            if (key && !value) {
                implyFail.push(origKey);
            }
        });

        if (implyFail.length) {
            var msg = 'Implications failed:\n';

            implyFail.forEach(function (key) {
                msg += ('  ' + key + ' -> ' + implied[key] + '\n');
            });

            fail(msg);
        }
        
        return argv;
    }
    
    function longest (xs) {
        return Math.max.apply(
            null,
            xs.map(function (x) { return x.length })
        );
    }
    
    return self;
};

// rebase an absolute path to a relative one with respect to a base directory
// exported for tests
exports.rebase = rebase;
function rebase (base, dir) {
    var ds = path.normalize(dir).split('/').slice(1);
    var bs = path.normalize(base).split('/').slice(1);
    
    for (var i = 0; ds[i] && ds[i] == bs[i]; i++);
    ds.splice(0, i); bs.splice(0, i);
    
    var p = path.normalize(
        bs.map(function () { return '..' }).concat(ds).join('/')
    ).replace(/\/$/,'').replace(/^$/, '.');
    return p.match(/^[.\/]/) ? p : './' + p;
};
