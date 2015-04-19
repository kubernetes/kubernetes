var expect = require('expect.js');
var helpers = require('../helpers');

describe('rc', function() {
    var tempDir = new helpers.TempDir();

    var rc = require('../../lib/util/rc');
    var defaults = require('../../lib/util/defaults');
    defaults.cwd = tempDir.path;

    tempDir.prepare({
        '.bowerrc': {
            key: 'value'
        },
        'child/.bowerrc': {
            key2: 'value2'
        },
        'child2/.bowerrc': {
            key: 'valueShouldBeOverwriteParent'
        },
        'child3/bower.json': {
            name: 'without-bowerrc'
        }
    });

    it('correctly reads .bowerrc files', function() {
        var config = rc('bower', defaults, tempDir.path);

        expect(config.key).to.eql('value');
        expect(config.key2).to.eql(undefined);
    });

    it('correctly reads .bowerrc files from child', function() {
        var config = rc('bower', defaults, tempDir.path + '/child/');

        expect(config.key).to.eql('value');
        expect(config.key2).to.eql('value2');
    });

    it('correctly reads .bowerrc files from child2', function() {
        var config = rc('bower', defaults, tempDir.path + '/child2/');

        expect(config.key).to.eql('valueShouldBeOverwriteParent');
        expect(config.key2).to.eql(undefined);
    });

    it('correctly reads .bowerrc files from child3', function() {
        var config = rc('bower', defaults, tempDir.path + '/child3/');

        expect(config.key).to.eql('value');
        expect(config.key2).to.eql(undefined);
    });
});
