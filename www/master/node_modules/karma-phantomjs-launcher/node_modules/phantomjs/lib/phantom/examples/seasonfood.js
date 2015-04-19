// Show BBC seasonal food list.

var cbfunc = function (data) {
    var list = data.query.results.results.result,
        names = ['January', 'February', 'March',
                 'April', 'May', 'June',
                 'July', 'August', 'September',
                 'October', 'November', 'December'];
    list.forEach(function (item) {
        console.log([item.name.replace(/\s/ig, ' '), ':',
                  names[item.atItsBestUntil], 'to',
                  names[item.atItsBestFrom]].join(' '));
    });
    phantom.exit();
};

var el = document.createElement('script');
el.src = 'http://query.yahooapis.com/v1/public/yql?q=SELECT%20*%20FROM%20bbc.goodfood.seasonal%3B&format=json&diagnostics=true&env=store%3A%2F%2Fdatatables.org%2Falltableswithkeys&callback=cbfunc';
document.body.appendChild(el);
