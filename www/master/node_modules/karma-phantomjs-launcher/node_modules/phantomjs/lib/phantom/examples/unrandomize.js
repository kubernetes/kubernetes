// Modify global object at the page initialization.
// In this example, effectively Math.random() always returns 0.42.

var page = require('webpage').create();

page.onInitialized = function () {
    page.evaluate(function () {
        Math.random = function() {
            return 42 / 100;
        };
    });
};

page.open('http://ariya.github.com/js/random/', function (status) {
    var result;
    if (status !== 'success') {
        console.log('Network error.');
    } else {
        console.log(page.evaluate(function () {
            return document.getElementById('numbers').textContent;
        }));
    }
    phantom.exit();
});
