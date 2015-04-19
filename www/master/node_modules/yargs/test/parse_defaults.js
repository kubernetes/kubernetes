var should = require('chai').should(),
    yargs = require('../');

describe('parse', function () {

    describe('defaults', function () {
        function checkNoArgs(argv, hasAlias) {
            it('should set defaults if no args', function() {
                var result = argv.parse([ ]);
                result.should.have.property('flag', true);
                if (hasAlias) {
                    result.should.have.property('f', true);
                }
            });
        }

        function checkExtraArg(argv, hasAlias) {
            it('should set defaults if one extra arg', function() {
                var result = argv.parse([ 'extra' ]);
                result.should.have.property('flag', true);
                result.should.have.property('_').and.deep.equal(['extra']);
                if (hasAlias) {
                    result.should.have.property('f', true);
                }
            });
        }

        function checkStringArg(argv, hasAlias) {
            it('should set defaults even if arg looks like a string', function() {
                var result = argv.parse([ '--flag', 'extra' ]);
                result.should.have.property('flag', true);
                result.should.have.property('_').and.deep.equal(['extra']);
                if (hasAlias) {
                    result.should.have.property('f', true);
                }
            });
        }

        describe('for options with aliases', function () {
            var args = yargs().options({
                flag : {
                    alias   : 'f',
                    default : true
                }
            });

            checkNoArgs(args, true);
            checkExtraArg(args, true);
            // This test case should fail, because we didn't specify that the
            // option is a boolean
            // checkStringArg(args, true);
        });

        describe('for typed options without aliases', function () {
            var args = yargs().options({
                flag : {
                    type    : 'boolean',
                    default : true
                }
            });

            checkNoArgs(args);
            checkExtraArg(args);
            checkStringArg(args);
        });

        describe('for typed options with aliases', function () {
            var args = yargs().options({
                flag : {
                    alias   : 'f',
                    type    : 'boolean',
                    default : true
                }
            });

            checkNoArgs(args, true);
            checkExtraArg(args, true);
            checkStringArg(args, true);
        });
    });

});
