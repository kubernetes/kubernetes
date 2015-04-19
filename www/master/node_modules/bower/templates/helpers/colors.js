var chalk = require('chalk');

var templateColors = [
    'yellow',
    'green',
    'cyan',
    'red',
    'white',
    'magenta'
];

function colors(Handlebars) {
    templateColors.forEach(function (color) {
        Handlebars.registerHelper(color, function (context) {
            return chalk[color](context.fn(this));
        });
    });
}

module.exports = colors;
