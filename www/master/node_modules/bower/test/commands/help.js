var expect = require('expect.js');
var helpers = require('../helpers');
var help = helpers.command('help');

describe('bower help', function () {

    it('correctly reads arguments', function() {
        expect(help.readOptions(['foo'])).to.eql(['foo']);
    });

    it('shows general help', function () {
        return helpers.run(help).spread(function(result) {
            expect(result.usage[0]).to.be.a('string');
            expect(result.commands).to.be.a('object');
            expect(result.options).to.be.a('object');
        });
    });

    var commands = [
        'home', 'info', 'init', 'install',
        'link', 'list', 'lookup', 'prune', 'register',
        'search', 'update', 'uninstall', 'version',
        'cache list', 'cache clean'
    ];

    commands.forEach(function(command) {
        it('shows help for ' + command + ' command', function() {
            return helpers.run(help, [command]).spread(function(result) {
                expect(result.command).to.be(command);
                expect(result.description).to.be.a('string');
                expect(result.usage[0]).to.be.a('string');
            });
        });
    });

    it('displays error for non-existing command', function() {
        return helpers.run(help, ['fuu']).fail(function(e) {
            expect(e.message).to.be('Unknown command: fuu');
            expect(e.command).to.be('fuu');
            expect(e.code).to.be('EUNKNOWNCMD');
        });
    });
});
