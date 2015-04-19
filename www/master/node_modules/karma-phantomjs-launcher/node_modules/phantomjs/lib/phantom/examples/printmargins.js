var page = require('webpage').create(),
    system = require('system');

if (system.args.length < 7) {
    console.log('Usage: printmargins.js URL filename LEFT TOP RIGHT BOTTOM');
    console.log('  margin examples: "1cm", "10px", "7mm", "5in"');
    phantom.exit(1);
} else {
    var address = system.args[1];
    var output = system.args[2];
    var marginLeft = system.args[3];
    var marginTop = system.args[4];
    var marginRight = system.args[5];
    var marginBottom = system.args[6];
    page.viewportSize = { width: 600, height: 600 };
    page.paperSize = {
        format: 'A4',
        margin: {
            left: marginLeft,
            top: marginTop,
            right: marginRight,
            bottom: marginBottom
        }
    };
    page.open(address, function (status) {
        if (status !== 'success') {
            console.log('Unable to load the address!');
        } else {
            window.setTimeout(function () {
                page.render(output);
                phantom.exit();
            }, 200);
        }
    });
}
