function sum(Handlebars) {
    Handlebars.registerHelper('sum', function (val1, val2) {
        return val1 + val2;
    });
}

module.exports = sum;
