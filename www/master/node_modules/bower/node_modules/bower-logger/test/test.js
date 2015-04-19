var expect = require('expect.js');
var EventEmitter = require('events').EventEmitter;
var Logger = require('../');

describe('Logger', function () {
    var logger;

    beforeEach(function () {
        logger = new Logger();
    });

    describe('.constructor', function () {
        it('should provide an instance of Logger', function () {
            expect(logger instanceof Logger).to.be(true);
        });

        it('should provide an instance of EventEmitter', function () {
            expect(logger instanceof EventEmitter).to.be(true);
        });

        it('should have prototype methods', function () {
            var methods = [
                    'intercept', 'pipe', 'geminate', 'log'
                ];

            methods.forEach(function (method) {
                expect(logger).to.have.property(method);
            });
        });
    });

    describe('events', function () {
        var logData = {
            foo: 'bar',
            baz: 'string'
        };

        it('should pass through {}', function (next) {
            logger.on('log', function (log) {
                expect(log.data).to.eql({});
                next();
            });
            logger.info();
        });

        it('should pass through logData', function (next) {
            logger.on('log', function (log) {
                expect(log.data).to.eql(logData);
                next();
            });
            logger.info('foo', 'message', logData);
        });

        it('should emit error event', function (next) {
            logger.on('log', function (log) {
                expect(log.level).to.eql('error');
                expect(log.id).to.eql('foo');
                expect(log.message).to.eql('error message');
                expect(log.data).to.eql({});
                next();
            });
            logger.error('foo', 'error message');
        });

        it('should emit conflict event', function (next) {
            logger.on('log', function (log) {
                expect(log.level).to.eql('conflict');
                expect(log.id).to.eql('foo');
                expect(log.message).to.eql('conflict message');
                expect(log.data).to.eql({});
                next();
            });
            logger.conflict('foo', 'conflict message');
        });

        it('should emit warn event', function (next) {
            logger.on('log', function (log) {
                expect(log.level).to.eql('warn');
                expect(log.id).to.eql('foo');
                expect(log.message).to.eql('warn message');
                expect(log.data).to.eql({});
                next();
            });
            logger.warn('foo', 'warn message');
        });

        it('should emit action event', function (next) {
            logger.on('log', function (log) {
                expect(log.level).to.eql('action');
                expect(log.id).to.eql('foo');
                expect(log.message).to.eql('action message');
                expect(log.data).to.eql({});
                next();
            });
            logger.action('foo', 'action message');
        });

        it('should emit info event', function (next) {
            logger.on('log', function (log) {
                expect(log.level).to.eql('info');
                expect(log.id).to.eql('foo');
                expect(log.message).to.eql('info message');
                expect(log.data).to.eql({});
                next();
            });
            logger.info('foo', 'info message');
        });

        it('should emit debug event', function (next) {
            logger.on('log', function (log) {
                expect(log.level).to.eql('debug');
                expect(log.id).to.eql('foo');
                expect(log.message).to.eql('debug message');
                expect(log.data).to.eql({});
                next();
            });
            logger.debug('foo', 'debug message');
        });
    });

    describe('.intercept', function () {
        it('should add the function and call it when a log occurs', function (next) {
            var called;
            var data = {
                'some': 'thing'
            };

            logger.intercept(function (log) {
                called = true;

                expect(log).to.eql({
                    level: 'warn',
                    id: 'foo',
                    message: 'bar',
                    data: data
                });
                expect(log.data).to.equal(data);
            });

            logger.log('warn', 'foo', 'bar', data);

            expect(called).to.be(true);
            next();
        });

        it('should call the interceptors by order before emitting the event', function (next) {
            var called = [];

            logger.intercept(function () {
                called.push(1);
            });
            logger.intercept(function () {
                called.push(2);
            });

            logger.log('warn', 'foo', 'bar');

            expect(called).to.eql([1, 2]);
            next();
        });

        it('should call the interceptors along the chain', function (next) {
            var called = [];
            var childLogger = logger.geminate();

            childLogger.intercept(function () {
                called.push(1);
            });
            logger.intercept(function () {
                called.push(3);
            });

            childLogger.on('log', function () {
                called.push(2);
            });
            logger.on('log', function () {
                called.push(4);
            });

            childLogger.log('warn', 'foo', 'bar');

            expect(called).to.eql([1, 2, 3, 4]);
            next();
        });
    });

    describe('.pipe', function () {
        it('should return the passed emitter', function () {
            var otherEmitter = new EventEmitter();
            expect(logger.pipe(otherEmitter)).to.equal(otherEmitter);
        });

        it('should pipe log events to another emitter', function (next) {
            var otherEmitter = new EventEmitter();
            var data = {
                'some': 'thing'
            };
            var piped;

            logger.pipe(otherEmitter);

            otherEmitter.on('log', function (log) {
                piped = true;
                expect(log).to.eql({
                    level: 'warn',
                    id: 'foo',
                    message: 'bar',
                    data: data
                });
            });

            logger.log('warn', 'foo', 'bar', data);

            expect(piped).to.be(true);
            next();
        });
    });

    describe('.geminate', function () {
        it('should return a new logger instance', function () {
            var newLogger = logger.geminate();

            expect(newLogger).to.be.an(Logger);
            expect(newLogger).to.be.an(EventEmitter);
            expect(newLogger).to.not.be.equal(logger);
        });

        it('should pipe the new logger events to the original logger', function (next) {
            var piped = [];
            var childLogger = logger.geminate();
            var data = {
                'some': 'thing'
            };

            childLogger.on('log', function (log) {
                piped.push(1);
                expect(log).to.eql({
                    level: 'warn',
                    id: 'foo',
                    message: 'bar',
                    data: data
                });
                expect(log.data).to.equal(data);
            });

            logger.on('log', function (log) {
                piped.push(2);
                expect(log).to.eql({
                    level: 'warn',
                    id: 'foo',
                    message: 'bar',
                    data: data
                });
                expect(log.data).to.equal(data);
            });

            childLogger.log('warn', 'foo', 'bar', data);
            expect(piped).to.eql([1, 2]);
            next();
        });
    });

    describe('.prompt', function () {
        it('should only allow calling the callback once', function () {
            var calls = 0;

            logger
            .once('prompt', function (prompts, callback) {
                callback({ prompt: 'bar' });
                callback({ prompt: 'foo' });
            })
            .prompt({
                type: 'input',
                message: 'foo'
            }, function () {
                calls += 1;
            });

            expect(calls).to.equal(1);
        });

        it('should accept a prompt', function (next) {
            logger
            .once('prompt', function (prompts, callback) {
                callback({
                    prompt: 'bar'
                });
            })
            .prompt({
                type: 'input',
                message: 'foo'
            }, function (err, answer) {
                expect(err).to.not.be.ok();
                expect(answer).to.equal('bar');
                next();
            });
        });

        it('should accept several prompts', function (next) {
            logger
            .once('prompt', function (prompts, callback) {
                callback({
                    foo: 'bar',
                    foz: 'baz'
                });
            })
            .prompt([
                {
                    name: 'foo',
                    type: 'input',
                    message: 'foo'
                },
                {
                    name: 'foz',
                    type: 'confirm',
                    message: 'foz'
                }
            ], function (err, answer) {
                expect(err).to.not.be.ok();
                expect(answer.foo).to.equal('bar');
                expect(answer.foz).to.equal('baz');

                logger
                .once('prompt', function (prompts, callback) {
                    callback({
                        foo: 'bar'
                    });
                })
                .prompt([
                    {
                        name: 'foo',
                        type: 'input',
                        message: 'foo'
                    }
                ], function (err, answer) {
                    expect(err).to.not.be.ok();
                    expect(answer.foo).to.equal('bar');
                    next();
                });
            });
        });

        it('should error on invalid prompt type', function (next) {
            logger.prompt({
                type: 'xxx',
                message: 'foo'
            }, function (err) {
                expect(err).to.be.an(Error);
                expect(err.code).to.be('ENOTSUP');
                next();
            });
        });

        it('should trim the answers', function (next) {
            logger
            .once('prompt', function (prompts, callback) {
                callback({
                    prompt: ' bar '
                });
            })
            .prompt({
                type: 'input',
                message: 'foo'
            }, function (err, answer) {
                expect(err).to.not.be.ok();
                expect(answer).to.equal('bar');
                next();
            });
        });

        it('should trim multiple response answers', function (next) {
            logger
            .once('prompt', function (prompts, callback) {
                callback({
                    prompt: [' bar ', ' foo', 'baz ']
                });
            })
            .prompt({
                type: 'checkbox',
                message: 'foo'
            }, function (err, answer) {
                expect(err).to.not.be.ok();
                expect(answer).to.eql(['bar', 'foo', 'baz']);
                next();
            });
        });

    });
});
