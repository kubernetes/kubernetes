var path = require('path');
var expect = require('expect.js');
var fs = require('fs');

var helpers = require('../helpers');
var uninstall = helpers.command('uninstall');

describe('bower uninstall', function () {

    var tempDir = new helpers.TempDir({
        'bower.json': {
            name: 'hello-world',
            dependencies: {
                'underscore': '*'
            }
        }
    });

    beforeEach(function() {
        tempDir.prepare();
    });

    var bowerJsonPath = path.join(tempDir.path, 'bower.json');

    function bowerJson() {
        return JSON.parse(fs.readFileSync(bowerJsonPath));
    }

    var config = {
        cwd: tempDir.path,
        interactive: true
    };

    it('correctly reads arguments', function() {
        expect(uninstall.readOptions(['jquery', '-S', '-D']))
        .to.eql([['jquery'], { save: true, saveDev: true }]);
    });

    it('correctly reads long arguments', function() {
        expect(uninstall.readOptions(['jquery', '--save', '--save-dev']))
        .to.eql([['jquery'], { save: true, saveDev: true }]);
    });

    it('does not remove anything from dependencies by default', function () {
        return helpers.run(uninstall, [['underscore'], undefined, config]).then(function () {
            expect(bowerJson().dependencies).to.eql({ 'underscore': '*' });
        });
    });

    it('removes dependency from bower.json if --save flag is used', function () {
        return helpers.run(uninstall, [['underscore'], {save: true}, config]).then(function () {
            expect(bowerJson().dependencies).to.eql({});
        });
    });

});
