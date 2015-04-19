var page = require('webpage').create(),
    system = require('system'),
    city,
    url;

city = 'Mountain View, California'; // default
if (system.args.length > 1) {
    city = Array.prototype.slice.call(system.args, 1).join(' ');
}
url = encodeURI('http://api.openweathermap.org/data/2.1/find/name?q=' + city);

console.log('Checking weather condition for', city, '...');

page.open(url, function(status) {
    var result, data;
    if (status !== 'success') {
        console.log('Error: Unable to access network!');
    } else {
        result = page.evaluate(function () {
            return document.body.innerText;
        });
        try {
            data = JSON.parse(result);
            data = data.list[0];
            console.log('');
            console.log('City:', data.name);
            console.log('Condition:', data.weather.map(function(entry) {
                return entry.main;
            }).join(', '));
            console.log('Temperature:', Math.round(data.main.temp - 273.15), 'C');
            console.log('Humidity:', Math.round(data.main.humidity), '%');
        } catch (e) {
            console.log('Error:', e.toString());
        }
    }
    phantom.exit();
});
