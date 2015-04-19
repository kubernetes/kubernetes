var expect = require('expect.js');
var helpers = require('../helpers');

var lookup = helpers.command('lookup');

describe('bower lookup', function () {

    var lookupWithResult = function (response) {
        return helpers.command('lookup', {
            'bower-registry-client': function() {
                return {
                    lookup: function(query, callback) {
                        if (query in response) {
                            callback(null, response[query]);
                        } else {
                            callback();
                        }
                    }
                };
            }
        });
    };

    it('correctly reads arguments', function() {
        expect(lookup.readOptions(['jquery']))
        .to.eql(['jquery']);
    });

    it('lookups package by name', function () {
        var lookup = lookupWithResult({ jquery: { url: 'http://jquery.org' } });

        return helpers.run(lookup, ['jquery']).spread(function(result) {
            expect(result).to.eql({
                name: 'jquery',
                url: 'http://jquery.org'
            });
        });
    });

    it('returns null if no package is found', function () {
        var lookup = lookupWithResult({ jquery: { url: 'http://jquery.org' } });

        return helpers.run(lookup, ['foobar']).spread(function(result) {
            expect(result).to.eql(null);
        });
    });

    it('returns null if called without argument', function () {
        var lookup = lookupWithResult({ jquery: { url: 'http://jquery.org' } });

        return helpers.run(lookup, []).spread(function(result) {
            expect(result).to.eql(null);
        });
    });
});
