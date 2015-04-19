var mout = require('mout');

function rpad(Handlebars) {
    Handlebars.registerHelper('rpad', function (context) {
        var hash = context.hash;
        var length = parseInt(hash.length, 10);
        var chr = hash.char;

        return mout.string.rpad(context.fn(this), length, chr);
    });
}

module.exports = rpad;
