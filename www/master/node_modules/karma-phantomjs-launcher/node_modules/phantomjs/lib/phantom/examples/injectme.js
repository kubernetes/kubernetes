// Use 'page.injectJs()' to load the script itself in the Page context

if ( typeof(phantom) !== "undefined" ) {
    var page = require('webpage').create();

    // Route "console.log()" calls from within the Page context to the main Phantom context (i.e. current "this")
    page.onConsoleMessage = function(msg) {
        console.log(msg);
    };
    
    page.onAlert = function(msg) {
        console.log(msg);
    };
    
    console.log("* Script running in the Phantom context.");
    console.log("* Script will 'inject' itself in a page...");
    page.open("about:blank", function(status) {
        if ( status === "success" ) {
            console.log(page.injectJs("injectme.js") ? "... done injecting itself!" : "... fail! Check the $PWD?!");
        }
        phantom.exit();
    });
} else {
    alert("* Script running in the Page context.");
}
