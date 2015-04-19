var should = require('chai').should(),
    Hash = require('hashish'),
    yargs = require('../');

describe('usage', function () {

    describe('demand options', function () {
        describe('using .demand()', function () {
            it ('should show an error along with the missing arguments on demand fail', function () {
                var r = checkUsage(function () {
                    return yargs('-x 10 -z 20'.split(' '))
                        .usage('Usage: $0 -x NUM -y NUM')
                        .demand(['x','y'])
                        .argv;
                });
                r.result.should.have.property('x', 10);
                r.result.should.have.property('z', 20);
                r.result.should.have.property('_').with.length(0);
                r.errors.join('\n').split(/\n+/).should.deep.equal([
                    'Usage: ./usage -x NUM -y NUM',
                    'Options:',
                    '  -x  [required]',
                    '  -y  [required]',
                    'Missing required arguments: y'
                ]);
                r.logs.should.have.length(0);
                r.exit.should.be.ok;
            });

            describe('using .require()', function() {
                it ('should show an error along with the missing arguments on demand fail', function () {
                    var r = checkUsage(function () {
                        return yargs('-x 10 -z 20'.split(' '))
                            .usage('Usage: $0 -x NUM -y NUM')
                            .require(['x','y'])
                            .argv;
                    });
                    r.result.should.have.property('x', 10);
                    r.result.should.have.property('z', 20);
                    r.result.should.have.property('_').with.length(0);
                    r.errors.join('\n').split(/\n+/).should.deep.equal([
                        'Usage: ./usage -x NUM -y NUM',
                        'Options:',
                        '  -x  [required]',
                        '  -y  [required]',
                        'Missing required arguments: y'
                    ]);
                    r.logs.should.have.length(0);
                    r.exit.should.be.ok;
                });
            });
        });

        it('should show an error along with a custom message on demand fail', function () {
            var r = checkUsage(function () {
                return yargs('-z 20'.split(' '))
                    .usage('Usage: $0 -x NUM -y NUM')
                    .demand(['x','y'], 'x and y are both required to multiply all the things')
                    .argv;
            });
            r.result.should.have.property('z', 20);
            r.result.should.have.property('_').with.length(0);
            r.errors.join('\n').split(/\n+/).should.deep.equal([
                'Usage: ./usage -x NUM -y NUM',
                'Options:',
                '  -x  [required]',
                '  -y  [required]',
                'Missing required arguments: x, y',
                'x and y are both required to multiply all the things'
            ]);
            r.logs.should.have.length(0);
            r.exit.should.be.ok;
        });

        it('should return valid values when demand passes', function () {
            var r = checkUsage(function () {
                return yargs('-x 10 -y 20'.split(' '))
                    .usage('Usage: $0 -x NUM -y NUM')
                    .demand(['x','y'])
                    .argv;
            });
            r.should.have.property('result');
            r.result.should.have.property('x', 10);
            r.result.should.have.property('y', 20)
            r.result.should.have.property('_').with.length(0);
            r.should.have.property('errors').with.length(0);
            r.should.have.property('logs').with.length(0);
            r.should.have.property('exit', false);
        });
    });

    it('should return valid values when check passes', function () {
        var r = checkUsage(function () {
            return yargs('-x 10 -y 20'.split(' '))
                .usage('Usage: $0 -x NUM -y NUM')
                .check(function (argv) {
                    if (!('x' in argv)) throw 'You forgot about -x';
                    if (!('y' in argv)) throw 'You forgot about -y';
                })
                .argv;
        });
        r.should.have.property('result');
        r.result.should.have.property('x', 10);
        r.result.should.have.property('y', 20);
        r.result.should.have.property('_').with.length(0);
        r.should.have.property('errors').with.length(0);
        r.should.have.property('logs').with.length(0);
        r.should.have.property('exit', false);
    });

    it('should display missing arguments when check fails with a thrown exception', function () {
        var r = checkUsage(function () {
            return yargs('-x 10 -z 20'.split(' '))
                .usage('Usage: $0 -x NUM -y NUM')
                .check(function (argv) {
                    if (!('x' in argv)) throw 'You forgot about -x';
                    if (!('y' in argv)) throw 'You forgot about -y';
                })
                .argv;
        });
        r.should.have.property('result');
        r.result.should.have.property('x', 10);
        r.result.should.have.property('z', 20);
        r.result.should.have.property('_').with.length(0);
        r.errors.join('\n').split(/\n+/).should.deep.equal([
            'Usage: ./usage -x NUM -y NUM',
            'You forgot about -y'
        ]);
        r.should.have.property('logs').with.length(0);
        r.should.have.property('exit').and.be.ok;
    });

    it('should display missing arguments when check fails with a return value', function () {
        var r = checkUsage(function () {
            return yargs('-x 10 -z 20'.split(' '))
                .usage('Usage: $0 -x NUM -y NUM')
                .check(function (argv) {
                    if (!('x' in argv)) return 'You forgot about -x';
                    if (!('y' in argv)) return 'You forgot about -y';
                })
                .argv;
        });
        r.should.have.property('result');
        r.result.should.have.property('x', 10);
        r.result.should.have.property('z', 20);
        r.result.should.have.property('_').with.length(0);
        r.should.have.property('logs').with.length(0);
        r.should.have.property('exit').and.be.ok;
        r.should.have.property('errors');
        r.errors.join('\n').split(/\n+/).should.deep.equal([
            'Usage: ./usage -x NUM -y NUM',
            'You forgot about -y'
        ]);
    });

    exports.checkFailReturn = function () {
        var r = checkUsage(function () {
            return yargs('-x 10 -z 20'.split(' '))
                .usage('Usage: $0 -x NUM -y NUM')
                .check(function (argv) {
                    if (!('x' in argv)) return 'You forgot about -x';
                    if (!('y' in argv)) return 'You forgot about -y';
                })
                .argv;
        });
        r.should.have.property('result');
        r.result.should.have.property('x', 10);
        r.result.should.have.property('z', 20);
        r.result.should.have.property('_').with.length(0);
        r.should.have.property('logs').with.length(0);
        r.should.have.property('exit').and.be.ok;
        r.should.have.property('errors');
        r.errors.join('\n').split(/\n+/).should.deep.equal([
            'Usage: ./usage -x NUM -y NUM',
            'You forgot about -y'
        ]);
    };

    it('should return a valid result when check condition passes', function () {
        function checker (argv) {
            return 'x' in argv && 'y' in argv;
        }
        var r = checkUsage(function () {
            return yargs('-x 10 -y 20'.split(' '))
                .usage('Usage: $0 -x NUM -y NUM')
                .check(checker)
                .argv;
        });
        r.should.have.property('result');
        r.result.should.have.property('x', 10);
        r.result.should.have.property('y', 20);
        r.result.should.have.property('_').with.length(0);
        r.should.have.property('errors').with.length(0);
        r.should.have.property('logs').with.length(0);
        r.should.have.property('exit', false);
    });

    it('should display a failed message when check condition fails', function () {
        function checker (argv) {
            return 'x' in argv && 'y' in argv;
        }
        var r = checkUsage(function () {
            return yargs('-x 10 -z 20'.split(' '))
                .usage('Usage: $0 -x NUM -y NUM')
                .check(checker)
                .argv;
        });
        r.should.have.property('result');
        r.result.should.have.property('x', 10);
        r.result.should.have.property('z', 20);
        r.result.should.have.property('_').with.length(0);
        r.should.have.property('logs').with.length(0);
        r.should.have.property('exit').and.be.ok;
        r.should.have.property('errors');
        r.errors.join('\n').split(/\n+/).join('\n').should.equal(
            'Usage: ./usage -x NUM -y NUM\n'
            + 'Argument check failed: ' + checker.toString()
        );
    });

    it('should return a valid result when demanding a count of non-hyphenated values', function () {
        var r = checkUsage(function () {
            return yargs('1 2 3 --moo'.split(' '))
                .usage('Usage: $0 [x] [y] [z] {OPTIONS}')
                .demand(3)
                .argv;
        });
        r.should.have.property('result');
        r.should.have.property('errors').with.length(0);
        r.should.have.property('logs').with.length(0);
        r.should.have.property('exit', false);
        r.result.should.have.property('_').and.deep.equal([1,2,3]);
        r.result.should.have.property('moo', true);
    });

    it('should return a failure message when not enough non-hyphenated arguments are found after a demand count', function () {
        var r = checkUsage(function () {
            return yargs('1 2 --moo'.split(' '))
                .usage('Usage: $0 [x] [y] [z] {OPTIONS}')
                .demand(3)
                .argv;
        });
        r.should.have.property('result');
        r.should.have.property('logs').with.length(0);
        r.should.have.property('exit').and.be.ok;
        r.result.should.have.property('_').and.deep.equal([1,2]);
        r.result.should.have.property('moo', true);
        r.should.have.property('errors');
        r.errors.join('\n').split(/\n+/).should.deep.equal([
            'Usage: ./usage [x] [y] [z] {OPTIONS}',
            'Not enough non-option arguments: got 2, need at least 3'
        ]);
    });

    it('should return a custom failure message when not enough non-hyphenated arguments are found after a demand count', function () {
        var r = checkUsage(function () {
            return yargs('src --moo'.split(' '))
                .usage('Usage: $0 [x] [y] [z] {OPTIONS} <src> <dest> [extra_files...]')
                .demand(2, 'src and dest files are both required')
                .argv;
        });
        r.should.have.property('result');
        r.should.have.property('logs').with.length(0);
        r.should.have.property('exit').and.be.ok;
        r.result.should.have.property('_').and.deep.equal(['src']);
        r.result.should.have.property('moo', true);
        r.should.have.property('errors');
        r.errors.join('\n').split(/\n+/).should.deep.equal([
            'Usage: ./usage [x] [y] [z] {OPTIONS} <src> <dest> [extra_files...]',
            'src and dest files are both required'
        ]);
    });

    it('should return a valid result when setting defaults for singles', function () {
        var r = checkUsage(function () {
            return yargs('--foo 50 --baz 70 --powsy'.split(' '))
                .default('foo', 5)
                .default('bar', 6)
                .default('baz', 7)
                .argv
            ;
        });
        r.should.have.property('result');
        r.result.should.have.property('foo', 50);
        r.result.should.have.property('bar', 6);
        r.result.should.have.property('baz', 70);
        r.result.should.have.property('powsy', true);
        r.result.should.have.property('_').with.length(0);
    });

    it('should return a valid result when default is set for an alias', function () {
        var r = checkUsage(function () {
            return yargs('')
                .alias('f', 'foo')
                .default('f', 5)
                .argv
            ;
        });
        r.should.have.property('result');
        r.result.should.have.property('f', 5);
        r.result.should.have.property('foo', 5);
        r.result.should.have.property('_').with.length(0);
    });

    it('should print a single line when failing and default is set for an alias', function() {
        var r = checkUsage(function() {
            return yargs('')
                .alias('f', 'foo')
                .default('f', 5)
                .demand(1)
                .argv
            ;
        });
        r.errors.join('\n').split(/\n+/).should.deep.equal([
            'Options:',
            '  -f, --foo  [default: 5]',
            'Not enough non-option arguments: got 0, need at least 1',
        ]);
    });

    it('should allow you to set default values for a hash of options', function () {
        var r = checkUsage(function () {
            return yargs('--foo 50 --baz 70'.split(' '))
                .default({ foo : 10, bar : 20, quux : 30 })
                .argv
            ;
        });
        r.should.have.property('result');
        r.result.should.have.property('_').with.length(0);
        r.result.should.have.property('foo', 50);
        r.result.should.have.property('baz', 70);
        r.result.should.have.property('bar', 20);
        r.result.should.have.property('quux', 30);
    });

    describe('required arguments', function () {
        describe('with options object', function () {
            it('should show a failure message if a required option is missing', function () {
                var r = checkUsage(function () {
                    var opts = {
                        foo: { description: 'foo option', alias: 'f', requiresArg: true },
                        bar: { description: 'bar option', alias: 'b', requiresArg: true }
                    };

                    return yargs('-f --bar 20'.split(' '))
                        .usage('Usage: $0 [options]', opts)
                        .argv;
                });
                r.should.have.property('result');
                r.result.should.have.property('_').with.length(0);
                r.should.have.property('errors');
                r.should.have.property('logs').with.length(0);
                r.should.have.property('exit').and.be.ok;
                r.errors.join('\n').split(/\n+/).should.deep.equal([
                    'Usage: ./usage [options]',
                    'Options:',
                    '  --foo, -f  foo option',
                    '  --bar, -b  bar option',
                    'Missing argument value: foo',
                ]);
            });

            it('should show a failure message if more than one required option is missing', function () {
                var r = checkUsage(function () {
                    var opts = {
                        foo: { description: 'foo option', alias: 'f', requiresArg: true },
                        bar: { description: 'bar option', alias: 'b', requiresArg: true }
                    };

                    return yargs('-f --bar'.split(' '))
                        .usage('Usage: $0 [options]', opts)
                        .argv;
                });
                r.should.have.property('result');
                r.result.should.have.property('_').with.length(0);
                r.should.have.property('errors');
                r.should.have.property('logs').with.length(0);
                r.should.have.property('exit').and.be.ok;
                r.errors.join('\n').split(/\n+/).should.deep.equal([
                    'Usage: ./usage [options]',
                    'Options:',
                    '  --foo, -f  foo option',
                    '  --bar, -b  bar option',
                    'Missing argument values: foo, bar',
                ]);
            });
        });

        describe('with requiresArg method', function () {
            it('should show a failure message if a required option is missing', function () {
                var r = checkUsage(function () {
                    var opts = {
                        foo: { description: 'foo option', alias: 'f' },
                        bar: { description: 'bar option', alias: 'b' }
                    };

                    return yargs('-f --bar 20'.split(' '))
                        .usage('Usage: $0 [options]', opts)
                        .requiresArg(['foo', 'bar'])
                        .argv;
                });
                r.should.have.property('result');
                r.result.should.have.property('_').with.length(0);
                r.should.have.property('errors');
                r.should.have.property('logs').with.length(0);
                r.should.have.property('exit').and.be.ok;
                r.errors.join('\n').split(/\n+/).should.deep.equal([
                    'Usage: ./usage [options]',
                    'Options:',
                    '  --foo, -f  foo option',
                    '  --bar, -b  bar option',
                    'Missing argument value: foo',
                ]);
            });
        });
    });

    context("with strict() option set", function () {
        it('should fail given an option argument that is not demanded', function () {
            var r = checkUsage(function () {
                opts = {
                    foo: { demand: 'foo option', alias: 'f' },
                    bar: { demand: 'bar option', alias: 'b' }
                };

                return yargs('-f 10 --bar 20 --baz 30'.split(' '))
                    .usage('Usage: $0 [options]', opts)
                    .strict()
                    .argv;
            });

            r.should.have.property('result');
            r.result.should.have.property('_').with.length(0);
            r.result.should.have.property('f', 10);
            r.result.should.have.property('foo', 10);
            r.result.should.have.property('b', 20);
            r.result.should.have.property('bar', 20);
            r.result.should.have.property('baz', 30);
            r.should.have.property('errors');
            r.errors.join('\n').split(/\n+/).should.deep.equal([
                'Usage: ./usage [options]',
                'Options:',
                '  --foo, -f  [required]',
                '  --bar, -b  [required]',
                'Unknown argument: baz',
            ]);
            r.should.have.property('logs').with.length(0);
            r.should.have.property('exit').and.be.ok;
        });

        it('should fail given an option argument without a corresponding description', function () {
            var r = checkUsage(function () {
                opts = {
                    foo: { description: 'foo option', alias: 'f' },
                    bar: { description: 'bar option', alias: 'b' }
                };

                return yargs('-f 10 --bar 20 --baz 30'.split(' '))
                    .usage('Usage: $0 [options]', opts)
                    .strict()
                    .argv;
            });

            r.should.have.property('result');
            r.result.should.have.property('_').with.length(0);
            r.result.should.have.property('f', 10);
            r.result.should.have.property('foo', 10);
            r.result.should.have.property('b', 20);
            r.result.should.have.property('bar', 20);
            r.result.should.have.property('baz', 30);
            r.should.have.property('errors');
            r.errors.join('\n').split(/\n+/).should.deep.equal([
                'Usage: ./usage [options]',
                'Options:',
                '  --foo, -f  foo option',
                '  --bar, -b  bar option',
                'Unknown argument: baz',
            ]);
            r.should.have.property('logs').with.length(0);
            r.should.have.property('exit').and.be.ok;
        });

        it('should fail given multiple option arguments without corresponding descriptions', function () {
            var r = checkUsage(function () {
                opts = {
                    foo: { description: 'foo option', alias: 'f' },
                    bar: { description: 'bar option', alias: 'b' }
                };

                return yargs('-f 10 --bar 20 --baz 30 -q 40'.split(' '))
                    .usage('Usage: $0 [options]', opts)
                    .strict()
                    .argv;
            });

            r.should.have.property('result');
            r.result.should.have.property('_').with.length(0);
            r.result.should.have.property('f', 10);
            r.result.should.have.property('foo', 10);
            r.result.should.have.property('b', 20);
            r.result.should.have.property('bar', 20);
            r.result.should.have.property('baz', 30);
            r.result.should.have.property('q', 40);
            r.should.have.property('errors');
            r.errors.join('\n').split(/\n+/).should.deep.equal([
                'Usage: ./usage [options]',
                'Options:',
                '  --foo, -f  foo option',
                '  --bar, -b  bar option',
                'Unknown arguments: baz, q',
            ]);
            r.should.have.property('logs').with.length(0);
            r.should.have.property('exit').and.be.ok;
        });

        it('should pass given option arguments with corresponding descriptions', function () {
            var r = checkUsage(function () {
                opts = {
                    foo: { description: 'foo option' },
                    bar: { description: 'bar option' }
                };

                return yargs('--foo 10 --bar 20'.split(' '))
                    .usage('Usage: $0 [options]', opts)
                    .strict()
                    .argv;
            });

            r.should.have.property('result');
            r.result.should.have.property('foo', 10);
            r.result.should.have.property('bar', 20)
            r.result.should.have.property('_').with.length(0);
            r.should.have.property('errors').with.length(0);
            r.should.have.property('logs').with.length(0);
            r.should.have.property('exit', false);
        });
    });

    it('should display example on fail', function () {
        var r = checkUsage(function () {
            return yargs('')
                .example("$0 something", "description")
                .example("$0 something else", "other description")
                .demand(['y'])
                .argv;
        });
        r.should.have.property('result');
        r.result.should.have.property('_').with.length(0);
        r.should.have.property('errors');
        r.should.have.property('logs').with.length(0);
        r.should.have.property('exit').and.be.ok;
        r.errors.join('\n').split(/\n+/).should.deep.equal([
            'Examples:',
            '  ./usage something         description',
            '  ./usage something else    other description',
            'Options:',
            '  -y  [required]',
            'Missing required arguments: y'
        ]);
    });

    describe('demand option with boolean flag', function () {
        describe('with demand option', function () {
            it('should report missing required arguments', function () {
                var r = checkUsage(function () {
                    return yargs('-y 10 -z 20'.split(' '))
                        .usage('Usage: $0 -x NUM [-y NUM]')
                        .options({
                            'x': { description: 'an option',      demand: true  },
                            'y': { description: 'another option', demand: false }
                        })
                        .argv;
                });
                r.result.should.have.property('y', 10);
                r.result.should.have.property('z', 20);
                r.result.should.have.property('_').with.length(0);
                r.errors.join('\n').split(/\n/).should.deep.equal([
                    'Usage: ./usage -x NUM [-y NUM]',
                    '',
                    'Options:',
                    '  -x  an option       [required]',
                    '  -y  another option',
                    '',
                    'Missing required arguments: x'
                ]);
                r.logs.should.have.length(0);
                r.exit.should.be.ok;
            });
        });

        describe('with required option', function () {
            it('should report missing required arguments', function () {
                var r = checkUsage(function () {
                    return yargs('-y 10 -z 20'.split(' '))
                        .usage('Usage: $0 -x NUM [-y NUM]')
                        .options({
                            'x': { description: 'an option',      required: true  },
                            'y': { description: 'another option', required: false }
                        })
                        .argv;
                });
                r.result.should.have.property('y', 10);
                r.result.should.have.property('z', 20);
                r.result.should.have.property('_').with.length(0);
                r.errors.join('\n').split(/\n/).should.deep.equal([
                    'Usage: ./usage -x NUM [-y NUM]',
                    '',
                    'Options:',
                    '  -x  an option       [required]',
                    '  -y  another option',
                    '',
                    'Missing required arguments: x'
                ]);
                r.logs.should.have.length(0);
                r.exit.should.be.ok;
            });
        });

        it('should not report missing required arguments when given an alias', function () {
            var r = checkUsage(function () {
                return yargs('-w 10'.split(' '))
                    .usage('Usage: $0 --width NUM [--height NUM]')
                    .options({
                        'width':  { description: 'Width',  alias: 'w', demand: true  },
                        'height': { description: 'Height', alias: 'h', demand: false }
                    })
                    .argv;
            });
            r.result.should.have.property('w', 10);
            r.result.should.have.property('_').with.length(0);
            r.should.have.property('errors').with.length(0);
            r.logs.should.have.length(0);
        });
    });

    describe('help option', function () {
        it('should display usage', function () {
            var r = checkUsage(function () {
                return yargs(['--help'])
                    .demand(['y'])
                    .help('help')
                    .argv;
            });
            r.should.have.property('result');
            r.result.should.have.property('_').with.length(0);
            r.should.have.property('errors');
            r.should.have.property('logs').with.length(1);
            r.should.have.property('exit').and.be.ok;
            r.logs.join('\n').split(/\n+/).should.deep.equal([
                'Options:',
                '  --help  Show help',
                '  -y                 [required]',
                ''
            ]);
        });

        it('should not show both dashed and camelCase aliases', function () {
            var r = checkUsage(function () {
                return yargs(['--help'])
                    .usage('Usage: $0 options')
                    .help('help')
                    .describe('some-opt', 'Some option')
                    .default('some-opt', 2)
                    .argv;
            });
            r.should.have.property('result');
            r.result.should.have.property('_').with.length(0);
            r.should.have.property('exit').and.be.ok;
            r.should.have.property('errors').with.length(0);
            r.should.have.property('logs');
            r.logs.join('\n').split(/\n+/).should.deep.equal([
                'Usage: ./usage options',
                'Options:',
                '  --help      Show help  ',
                '  --some-opt  Some option  [default: 2]',
                ''
            ]);
        });
    });

    describe('version option', function () {
        it('should display version', function () {
            var r = checkUsage(function () {
                return yargs(['--version'])
                    .version('1.0.1', 'version', 'Show version number')
                    .argv;
            });
            r.should.have.property('result');
            r.result.should.have.property('_').with.length(0);
            r.should.have.property('errors');
            r.should.have.property('logs').with.length(1);
            r.should.have.property('exit').and.be.ok;
            r.logs.join('\n').split(/\n+/).should.deep.equal([
                '1.0.1'
            ]);
        });
    });

    describe('showHelpOnFail', function () {
        it('should display user supplied message', function () {
            var opts = {
                foo: { desc: 'foo option', alias: 'f' },
                bar: { desc: 'bar option', alias: 'b' }
            };

            var r = checkUsage(function () {
                return yargs(['--foo'])
                    .usage('Usage: $0 [options]')
                    .options(opts)
                    .demand(['foo', 'bar'])
                    .showHelpOnFail(false, "Specify --help for available options")
                    .argv;
            });
            r.should.have.property('result');
            r.result.should.have.property('_').with.length(0);
            r.should.have.property('errors');
            r.should.have.property('logs').with.length(0);
            r.should.have.property('exit').and.be.ok;
            r.errors.join('\n').split(/\n/).should.deep.equal([
                'Missing required arguments: bar',
                '',
                'Specify --help for available options'
            ]);
        });
    });

    it('should succeed when rebase', function () {
        yargs.rebase('/home/chevex', '/home/chevex/foo/bar/baz').should.equal('./foo/bar/baz');
        yargs.rebase('/home/chevex/foo/bar/baz', '/home/chevex').should.equal('../../..');
        yargs.rebase('/home/chevex/foo', '/home/chevex/pow/zoom.txt').should.equal('../pow/zoom.txt');
    });

    function checkUsage (f) {

        var exit = false;

        process._exit = process.exit;
        process._env = process.env;
        process._argv = process.argv;
        process.stdout._write = process.stdout.write;

        process.exit = function () { exit = true };
        process.env = Hash.merge(process.env, { _ : 'node' });
        process.argv = [ './usage' ];
        process.stdout.write = function (msg) { logs.push(msg) };

        var errors = [];
        var logs = [];

        console._error = console.error;
        console.error = function (msg) { errors.push(msg) };
        console._log = console.log;
        console.log = function (msg) { logs.push(msg) };

        var result = f();

        process.exit = process._exit;
        process.env = process._env;
        process.argv = process._argv;
        process.stdout.write = process.stdout._write;

        console.error = console._error;
        console.log = console._log;

        return {
            errors : errors,
            logs : logs,
            exit : exit,
            result : result
        };
    };

});
