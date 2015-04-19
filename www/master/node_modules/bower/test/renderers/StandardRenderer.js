// Simulate wide terminal
process.stdout.columns = 130;

var expect = require('chai').expect;
var helpers = require('../helpers');
var multiline = require('multiline').stripIndent;

var StandardRenderer = helpers.require('lib/renderers/StandardRenderer');

describe('StandardRenderer', function () {

    it('logs generic simple message', function () {
        return helpers.capture(function() {
            var renderer = new StandardRenderer();
            renderer.log({
                id: 'foobar',
                message: 'hello world'
            });
        }).spread(function(stdout, stderr) {
            expect(stdout).to.eq(multiline(function(){/*
                bower foobar        hello world

            */}));
        });
    });

    it('logs simple error', function () {
        return helpers.capture(function() {
            var renderer = new StandardRenderer();
            renderer.error({
                code: 'EFOOBAR',
                message: 'Hello error'
            });
        }).spread(function(stdout, stderr) {
            expect(stderr).to.eq(multiline(function(){/*
                bower EFOOBAR       Hello error

            */}));
        });
    });

    it('logs error with details', function () {
        return helpers.capture(function() {
            var renderer = new StandardRenderer();
            renderer.error({
                code: 'EFOOBAR',
                message: 'Hello error',
                details: '  Some awesome details\nMultiline!    '
            });
        }).spread(function(stdout, stderr) {
            expect(stderr).to.eq(multiline(function(){/*
                bower EFOOBAR       Hello error

                Additional error details:
                Some awesome details
                Multiline!

            */}));
        });
    });

    it('logs system details in verbose mode', function () {
        return helpers.capture(function() {
            var renderer = new StandardRenderer(undefined, { verbose: true });
            renderer.error({
                code: 'EFOOBAR',
                message: 'Hello error',
                details: '  Some awesome details\nMultiline!    '
            });
        }).spread(function(stdout, stderr) {
            expect(stderr).to.match(new RegExp(multiline(function(){/*
                System info:
                Bower version: [^\r\n]+
                Node version: [^\r\n]+
                OS: [^\r\n]+

            */})));
        });
    });

    it('logs stack trace in verbose mode', function () {
        return helpers.capture(function() {
            var renderer = new StandardRenderer(undefined, { verbose: true });
            renderer.error({
                code: 'EFOOBAR',
                message: 'Hello error',
                details: '  Some awesome details\nMultiline!    ',
                stack: [
                    './one.js:1',
                    './two.js:2'
                ]
            });
        }).spread(function(stdout, stderr) {
            expect(stderr).to.string(multiline(function(){/*
                Stack trace:
                ./one.js:1
                ./two.js:2

            */}));
        });
    });

    it('logs console trace in verbose mode', function () {
        return helpers.capture(function() {
            var renderer = new StandardRenderer(undefined, { verbose: true });
            renderer.error({
                code: 'EFOOBAR',
                message: 'Hello error',
                details: '  Some awesome details\nMultiline!    '
            });
        }).spread(function(stdout, stderr) {
            expect(stderr).to.match(new RegExp(multiline(function(){/*
                Console trace:
                Error
                    at StandardRenderer.error \(.+?\)

            */})));
        });
    });

    it('outputs checkout command log', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer();
            renderer.log({
                id: 'checkout',
                origin: 'jquery#master',
                message: 'foobar'
            });
        }).spread(function(stdout, stderr) {
            expect(stdout).to.equal(multiline(function(){/*
                bower checkout      jquery#foobar

            */}));
        });
    });

    it('outputs full progress for wide command', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer('install');
            renderer.log({
                id: 'progress',
                origin: 'jquery#master',
                message: 'foobar'
            });
        }).spread(function(stdout, stderr) {
            expect(stdout).to.equal(multiline(function(){/*
                bower jquery#master           progress foobar

            */}));
        });
    });

    it('outputs full progress for narrow command', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer('help');
            renderer.log({
                id: 'progress',
                origin: 'jquery#master',
                message: 'foobar'
            });
        }).spread(function(stdout, stderr) {
            expect(stdout).to.equal(multiline(function(){/*
                bower progress      jquery#master foobar

            */}));
        });
    });

    it('outputs extract log just as progress log', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer('install');
            renderer.log({
                id: 'extract',
                origin: 'jquery#master',
                message: 'foobar'
            });
        }).spread(function(stdout, stderr) {
            expect(stdout).to.equal(multiline(function(){/*
                bower jquery#master            extract foobar

            */}));
        });
    });

    it('outputs incompatible log with suitable package', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer();
            renderer.log({
                id: 'incompatible',
                data: {
                    resolution: '~0.1.1',
                    suitable: {
                        pkgMeta: {
                            _release: '0.1.2'
                        },
                        endpoint: {
                            name: 'foobar'
                        }
                    },
                    picks: [
                        {
                            pkgMeta: {
                                _release: '0.0.0'
                            },
                            endpoint: {
                                name: 'fizfuz',
                                target: '~0.0.0'
                            },
                            dependants: [
                                {
                                    pkgMeta: {
                                        _release: 'release1'
                                    },
                                    endpoint: {
                                        name: 'dependant1'
                                    }
                                },
                                {
                                    pkgMeta: {
                                        _release: 'release2'
                                    },
                                    endpoint: {
                                        name: 'dependant2'
                                    }
                                }
                            ]
                        },
                        {
                            endpoint: {
                                name: 'fizfuz2'
                            },
                            dependants: [
                                {
                                    pkgMeta: {
                                        // no release
                                    },
                                    endpoint: {
                                        name: 'jquery2'
                                    }
                                }
                            ]
                        }
                    ]
                }
            });
        }).spread(function(stdout, stderr) {
            expect(stdout).to.equal(multiline(function(){/*

                Please note that,
                    dependant1#release1, dependant2#release2 depends on fizfuz#~0.0.0 which resolved to fizfuz#0.0.0
                    jquery2 depends on fizfuz2#
                Resort to using foobar#~0.1.1 which resolved to foobar#0.1.2
                Code incompatibilities may occur.


            */}));
        });
    });

    it('outputs solver log without suitable package', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer();
            renderer.log({
                id: 'solved',
                data: {
                    resolution: '~0.1.1',
                    picks: [
                        {
                            pkgMeta: {
                                _release: '0.0.0'
                            },
                            endpoint: {
                                name: 'fizfuz',
                                target: '~0.0.0'
                            },
                            dependants: [
                                {
                                    pkgMeta: {
                                        _release: 'release1'
                                    },
                                    endpoint: {
                                        name: 'dependant1'
                                    }
                                },
                                {
                                    pkgMeta: {
                                        _release: 'release2'
                                    },
                                    endpoint: {
                                        name: 'dependant2'
                                    }
                                }
                            ]
                        },
                        {
                            endpoint: {
                                name: 'fizfuz2'
                            },
                            dependants: [
                                {
                                    pkgMeta: {
                                        // no release
                                    },
                                    endpoint: {
                                        name: 'jquery2'
                                    }
                                }
                            ]
                        }
                    ]
                }
            });
        }).spread(function(stdout, stderr) {
            expect(stdout).to.equal(multiline(function(){/*

                Unable to find a suitable version for , please choose one:
                    1) fizfuz#~0.0.0 which resolved to 0.0.0 and is required by dependant1#release1, dependant2#release2
                    2) fizfuz2# and is required by jquery2

                Prefix the choice with ! to persist it to bower.json


            */}));
        });
    });

    it('outputs json log', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer();
            renderer.log({
                id: 'json',
                data: {
                    json: {
                        foo: 'bar',
                        fiz: {
                            fuz: 'faz'
                        }
                    }
                }
            });
        }).spread(function(stdout, stderr) {
            expect(stdout).to.equal(multiline(function(){/*

                {
                  foo: 'bar',
                  fiz: {
                    fuz: 'faz'
                  }
                }


            */}));
        });
    });

    it('outputs cached entry log', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer('install');
            renderer.log({
                id: 'cached-entry',
                origin: 'origin',
                message: 'message'
            });
        }).spread(function(stdout, stderr) {
            expect(stdout).to.equal(multiline(function(){/*
                bower origin                    cached message

            */}));
        });
    });

    it('adjusts whitespace when package id too long', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer('install', {});
            renderer.log({
                id: 'generic',
                origin: 'short-origin',
                message: 'message'
            });

            renderer.log({
                id: 'generic',
                origin: 'very-very-long-origin-string',
                message: 'message'
            });

            renderer.log({
                id: 'generic',
                origin: 'short-origin',
                message: 'message'
            });
        }).spread(function(stdout, stderr) {
            expect(stdout).to.equal(multiline(function(){/*
                bower short-origin             generic message
                bower very-very-long-origin-string          generic message
                bower short-origin                          generic message

            */}));
        });
    });

    it('outputs install command log', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer('install', {
                cwd: '/tmp'
            });

            renderer.end([
                {
                    canonicalDir: '/tmp/components/jquery',
                    pkgMeta: {
                        _release: '0.1.2'
                    },
                    endpoint: {
                        name: 'jquery'
                    }
                },
                {
                    canonicalDir: '/tmp/components/jquery',
                    pkgMeta: {
                        version: '0.1.2'
                    },
                    endpoint: {
                        name: 'jquery'
                    }
                },
                {
                    canonicalDir: '/tmp/components/jquery',
                    pkgMeta: {
                        _release: '0.1.2'
                    },
                    endpoint: {
                        name: 'jquery'
                    },
                    missing: true
                },
                {
                    canonicalDir: '/tmp/components/jquery',
                    pkgMeta: {
                        _release: '0.1.2'
                    },
                    endpoint: {
                        name: 'jquery'
                    },
                    different: true
                },
                {
                    canonicalDir: '/tmp/components/jquery',
                    pkgMeta: {
                        _release: '0.1.2'
                    },
                    endpoint: {
                        name: 'jquery'
                    },
                    linked: true
                },
                {
                    canonicalDir: '/tmp/components/jquery',
                    pkgMeta: {
                        _release: '0.1.2'
                    },
                    endpoint: {
                        name: 'jquery',
                        target: '~0.1.2'
                    },
                    incompatible: true
                },
                {
                    canonicalDir: '/tmp/components/jquery',
                    pkgMeta: {
                        _release: '0.1.2'
                    },
                    endpoint: {
                        name: 'jquery',
                        target: '~0.1.2'
                    },
                    extraneous: true
                },
                {
                    canonicalDir: '/tmp/components/jquery',
                    pkgMeta: {
                        _release: '0.1.2'
                    },
                    endpoint: {
                        name: 'jquery',
                        target: '~0.1.2'
                    },
                    update: {
                        target: '0.1.5',
                        latest: '0.2.0'
                    }
                },
                {
                    canonicalDir: '/tmp/components/jquery',
                    pkgMeta: {
                        _release: '0.1.2'
                    },
                    endpoint: {
                        name: 'jquery'
                    },
                    dependencies: {
                        angular: {
                            canonicalDir: '/tmp/components/angular',
                            pkgMeta: {
                                _release: '0.1.3'
                            },
                            endpoint: {
                                name: 'angular'
                            }
                        },
                        ember: {
                            canonicalDir: '/tmp/components/ember',
                            pkgMeta: {
                                _release: '0.2.3'
                            },
                            endpoint: {
                                name: 'ember'
                            },
                            dependencies: {
                                // Should be ingored (only one level)
                                react: {
                                    canonicalDir: '/tmp/components/react',
                                    pkgMeta: {
                                        _release: '0.2.3'
                                    },
                                    endpoint: {
                                        name: 'react'
                                    }
                                }
                            }
                        }
                    }
                }
            ]);
        }).spread(function(stdout, stderr) {
            if (helpers.isWin()) {
                expect(stdout).to.equal(multiline(function(){/*

                    jquery#0.1.2 components\jquery

                    jquery#0.1.2 components\jquery

                    jquery components\jquery not installed

                    jquery#0.1.2 components\jquery different

                    jquery#0.1.2 components\jquery linked

                    jquery#0.1.2 components\jquery incompatible with ~0.1.2

                    jquery#0.1.2 components\jquery extraneous

                    jquery#0.1.2 components\jquery (0.1.5 available, latest is 0.2.0)

                    jquery#0.1.2 components\jquery
                    ├── angular#0.1.3
                    └── ember#0.2.3

                */}));
            } else {
                expect(stdout).to.equal(multiline(function(){/*

                    jquery#0.1.2 components/jquery

                    jquery#0.1.2 components/jquery

                    jquery components/jquery not installed

                    jquery#0.1.2 components/jquery different

                    jquery#0.1.2 components/jquery linked

                    jquery#0.1.2 components/jquery incompatible with ~0.1.2

                    jquery#0.1.2 components/jquery extraneous

                    jquery#0.1.2 components/jquery (0.1.5 available, latest is 0.2.0)

                    jquery#0.1.2 components/jquery
                    ├── angular#0.1.3
                    └── ember#0.2.3

                */}));
            }
        });
    });

    it('outputs short info command log', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer('info', {});
            renderer.end({
                version: '1.2.3'
            });
        }).spread(function(stdout, stderr) {
            expect(stdout).to.equal(multiline(function(){/*

                {
                  version: '1.2.3'
                }

            */}));
        });
    });

    it('outputs full info command log', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer('info', {});
            renderer.end({
                name: 'foo',
                latest: {
                    version: '1.2.3'
                },
                versions: [
                    '1.2.0',
                    '1.2.1',
                    '1.2.2'
                ]
            });
        }).spread(function(stdout, stderr) {
            expect(stdout).to.equal(multiline(function(){/*

                {
                  version: '1.2.3'
                }

                Available versions:
                  - 1.2.0
                  - 1.2.1
                  - 1.2.2
                You can request info for a specific version with 'bower info foo#<version>'

            */}));
        });
    });

    it('outputs lookup command log', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer('lookup', {});
            renderer.end({
                name: 'bower',
                url: 'http://bower.io'
            });
            renderer.end({
                name: 'bower'
            });
        }).spread(function(stdout, stderr) {
            expect(stdout).to.equal(multiline(function(){/*
                bower http://bower.io
                Package not found.

            */}));
        });
    });

    it('outputs link command log', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer('link', { cwd: '/tmp' });
            renderer.end({
                src: './foo',
                dst: './bar',
                installed: [{
                    canonicalDir: '/tmp/components/jquery',
                    pkgMeta: {
                        _release: '0.1.2'
                    },
                    endpoint: {
                        name: 'jquery'
                    }
                }]
            });
        }).spread(function(stdout, stderr) {
            if (helpers.isWin()) {
                expect(stdout).to.equal(multiline(function(){/*
                    bower                    link ./bar > ./foo

                    jquery#0.1.2 components\jquery

                */}));
            } else {
                expect(stdout).to.equal(multiline(function(){/*
                    bower                    link ./bar > ./foo

                    jquery#0.1.2 components/jquery

                */}));
            }
        });
    });

    it('outputs search command log', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer('search');
            renderer.end([
                {
                    name: 'jquery',
                    url: 'http://jquery.io'
                },
                {
                    name: 'bower',
                    url: 'http://bower.io'
                }
            ]);
        }).spread(function(stdout, stderr) {
            expect(stdout).to.equal(multiline(function(){/*
                Search results:

                    jquery http://jquery.io
                    bower http://bower.io

            */}));
        });
    });

    it('outputs register command log', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer('register');
            renderer.end({
                name: 'jquery',
                url: 'http://jquery.io'
            });
        }).spread(function(stdout, stderr) {
            expect(stdout).to.equal(multiline(function(){/*

                Package jquery registered successfully!
                All valid semver tags on http://jquery.io will be available as versions.
                To publish a new version, just release a valid semver tag.

                Run bower info jquery to list the available versions.

            */}));
        });
    });

    it('outputs cache list command log', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer('cache list');
            renderer.end([
                {
                    pkgMeta: {
                        name: 'awesome-jquery',
                        _target: '0.1.1',
                        _source: 'jquery'
                    }
                }
            ]);
        }).spread(function(stdout, stderr) {
            expect(stdout).to.equal(multiline(function(){/*
                awesome-jquery=jquery#0.1.1

            */}));
        });
    });

    it('outputs help command log', function() {
        return helpers.capture(function() {
            var renderer = new StandardRenderer('help');
            renderer.end({
                'command': 'uninstall',
                'description': 'Uninstalls a package locally from your bower_components directory',
                'usage': [
                    'uninstall <name> [<name> ..] [<options>]'
                ],
                'options': [
                    {
                        'shorthand':   '-h',
                        'flag':        '--help',
                        'description': 'Show this help message'
                    },
                    {
                        'shorthand':   '-S',
                        'flag':        '--save',
                        'description': 'Remove uninstalled packages from the project\'s bower.json dependencies'
                    },
                    {
                        'shorthand':   '-D',
                        'flag':        '--save-dev',
                        'description': 'Remove uninstalled packages from the project\'s bower.json devDependencies'
                    }
                ]
            });
        }).spread(function(stdout, stderr) {
            expect(stdout).to.equal(multiline(function(){/*

                Usage:

                    bower uninstall <name> [<name> ..] [<options>]
                Options:

                    -h, --help              Show this help message
                    -S, --save              Remove uninstalled packages from the project's bower.json dependencies
                    -D, --save-dev          Remove uninstalled packages from the project's bower.json devDependencies
                    Additionally all global options listed in 'bower help' are available

                Description:

                    Uninstalls a package locally from your bower_components directory

            */}));
        });
    });
});
