var Q = require('q');
var expect = require('expect.js');
var helpers = require('../helpers');

var search = helpers.command('search');

describe('bower search', function () {

    it('correctly reads arguments', function() {
        expect(search.readOptions(['jquery']))
        .to.eql(['jquery']);
    });

    it('searches for single repository', function () {
        return Q.Promise(function(resolve) {
            var search = helpers.command('search', {
                'bower-registry-client': function() {
                    return {
                        search: resolve
                    };
                }
            });

            helpers.run(search, ['jquery'], {});
        }).then(function(query) {
            expect(query).to.be('jquery');
        });
    });

    it('lists all repositories if no query given', function () {
        return Q.Promise(function(resolve) {
            var search = helpers.command('search', {
                'bower-registry-client': function() {
                    return {
                        list: resolve
                    };
                }
            });

            helpers.run(search, [], {});
        });
    });

});
