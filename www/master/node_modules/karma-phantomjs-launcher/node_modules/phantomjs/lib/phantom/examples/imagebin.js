// Upload an image to imagebin.org

var page = require('webpage').create(),
    system = require('system'),
    fname;

if (system.args.length !== 2) {
    console.log('Usage: imagebin.js filename');
    phantom.exit(1);
} else {
    fname = system.args[1];
    page.open("http://imagebin.org/index.php?page=add", function () {
        page.uploadFile('input[name=image]', fname);
        page.evaluate(function () {
            document.querySelector('input[name=nickname]').value = 'phantom';
            document.querySelector('input[name=disclaimer_agree]').click()
            document.querySelector('form').submit();
        });
        window.setTimeout(function () {
            phantom.exit();
        }, 3000);
    });
}
