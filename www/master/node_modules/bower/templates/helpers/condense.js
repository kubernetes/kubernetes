var mout = require('mout');
var leadLinesRegExp = /^\r?\n/;
var multipleLinesRegExp = /\r?\n(\r?\n)+/mg;

function condense(Handlebars) {
    Handlebars.registerHelper('condense', function (context) {
        var str = context.fn(this);

        // Remove multiple lines
        str = str.replace(multipleLinesRegExp, '$1');

        // Remove leading new lines (while keeping indentation)
        str = str.replace(leadLinesRegExp, '');

        // Remove trailing whitespaces (including new lines);
        str = mout.string.rtrim(str);

        return str;
    });
}

module.exports = condense;
