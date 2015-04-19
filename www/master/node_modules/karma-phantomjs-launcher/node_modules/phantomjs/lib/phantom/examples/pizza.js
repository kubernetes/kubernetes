// Find pizza in Mountain View using Yelp

var page = require('webpage').create(),
    url = 'http://lite.yelp.com/search?find_desc=pizza&find_loc=94040&find_submit=Search';

page.open(url, function (status) {
    if (status !== 'success') {
        console.log('Unable to access network');
    } else {
        var results = page.evaluate(function() {
            var list = document.querySelectorAll('address'), pizza = [], i;
            for (i = 0; i < list.length; i++) {
                pizza.push(list[i].innerText);
            }
            return pizza;
        });
        console.log(results.join('\n'));
    }
    phantom.exit();
});
