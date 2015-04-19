var path = require('path');
var fs = require('graceful-fs');
var Handlebars = require('handlebars');
var mout = require('mout');
var helpers = require('../../templates/helpers');

var templatesDir = path.resolve(__dirname, '../../templates');
var cache = {};

// Register helpers
mout.object.forOwn(helpers, function (register) {
    register(Handlebars);
});

function render(name, data, escape) {
    var contents;

    // Check if already compiled
    if (cache[name]) {
        return cache[name](data);
    }

    // Otherwise, read the file, compile and cache
    contents = fs.readFileSync(path.join(templatesDir, name)).toString();
    cache[name] = Handlebars.compile(contents, {
        noEscape: !escape
    });

    // Call the function again
    return render(name, data, escape);
}

function exists(name) {
    if (cache[name]) {
        return true;
    }

    return fs.existsSync(path.join(templatesDir, name));
}

module.exports = {
    render: render,
    exists: exists
};
