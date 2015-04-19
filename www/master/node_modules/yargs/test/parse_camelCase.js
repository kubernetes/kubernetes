var should = require('chai').should(),
    yargs = require('../');

describe('parse', function () {

    function runTests (yargs, strict) {

        if (!strict) {
            // Skip this test in strict mode because this option is not specified
            it('should provide options with dashes as camelCase properties', function () {
                var result = yargs()
                    .parse([ '--some-option' ]);

                result.should.have.property('some-option').that.is.a('boolean').and.is.true;
                result.should.have.property('someOption' ).that.is.a('boolean').and.is.true;
            });
        }

        it('should provide count options with dashes as camelCase properties', function () {
            var result = yargs()
                .option('some-option', {
                    describe : 'some option',
                    type     : 'count'
                })
                .parse([ '--some-option', '--some-option', '--some-option' ]);

            result.should.have.property('some-option', 3);
            result.should.have.property('someOption' , 3);
        });

        it('should provide options with dashes and aliases as camelCase properties', function () {
            var result = yargs()
                .option('some-option', {
                    alias    : 'o',
                    describe : 'some option'
                })
                .parse([ '--some-option' ]);

            result.should.have.property('some-option').that.is.a('boolean').and.is.true;
            result.should.have.property('someOption' ).that.is.a('boolean').and.is.true;
        });

        it('should provide defaults of options with dashes as camelCase properties', function() {
            var result = yargs()
                .option('some-option', {
                    describe : 'some option',
                    default  : 'asdf'
                })
                .parse([ ]);

            result.should.have.property('some-option', 'asdf');
            result.should.have.property('someOption' , 'asdf');
        });

        it('should provide aliases of options with dashes as camelCase properties', function() {
            var result = yargs()
                .option('some-option', {
                    alias    : 'o',
                    describe : 'some option',
                    default  : 'asdf'
                })
                .parse([ ]);

            result.should.have.property('o', 'asdf');
            result.should.have.property('some-option', 'asdf');
            result.should.have.property('someOption' , 'asdf');
        });

        it('should provide aliases of options with dashes as camelCase properties', function() {
            var result = yargs()
                .option('o', {
                    alias    : 'some-option',
                    describe : 'some option',
                    default  : 'asdf'
                })
                .parse([ ]);

            result.should.have.property('o', 'asdf');
            result.should.have.property('some-option', 'asdf');
            result.should.have.property('someOption' , 'asdf');
        });

        it('should provide aliases with dashes as camelCase properties', function() {
            var result = yargs()
                .option('o', {
                    alias    : 'some-option',
                    describe : 'some option'
                })
                .parse([ '--some-option', 'val' ]);

            result.should.have.property('o'          ).that.is.a('string').and.equals('val');
            result.should.have.property('some-option').that.is.a('string').and.equals('val');
            result.should.have.property('someOption' ).that.is.a('string').and.equals('val');
        });

    }

    describe('dashes and camelCase', function () {
        runTests(function() {
            return yargs();
        });
    });

    describe('dashes and camelCase (strict)', function () {
        runTests(function() {
            // Special handling for failure messages, because normally a
            // failure calls process.exit(1);
            return yargs().strict().fail(function(msg) {
                throw new Error(msg);
            });
        }, true);

        // See https://github.com/chevex/yargs/issues/31

        it('should not fail when options with defaults are missing', function () {
            var result = yargs()
                .fail(function(msg) {
                    throw new Error(msg);
                })
                .option('some-option', {
                    describe : 'some option',
                    default  : 80
                })
                .strict()
                .parse([ ]);
        });
    });

});
