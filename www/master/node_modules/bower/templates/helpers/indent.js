var mout = require('mout');

function indent(Handlebars) {
    Handlebars.registerHelper('indent', function (context) {
        var hash = context.hash;
        var indentStr = mout.string.repeat(' ', parseInt(hash.level, 10));

        return context.fn(this).replace(/\n/g, '\n' + indentStr);
    });
}

module.exports = indent;
