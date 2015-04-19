// The purpose of this is to show how and when events fire, considering 5 steps
// happening as follows:
//
//      1. Load URL
//      2. Load same URL, but adding an internal FRAGMENT to it
//      3. Click on an internal Link, that points to another internal FRAGMENT
//      4. Click on an external Link, that will send the page somewhere else
//      5. Close page
//
// Take particular care when going through the output, to understand when
// things happen (and in which order). Particularly, notice what DOESN'T
// happen during step 3.
//
// If invoked with "-v" it will print out the Page Resources as they are
// Requested and Received.
//
// NOTE.1: The "onConsoleMessage/onAlert/onPrompt/onConfirm" events are
// registered but not used here. This is left for you to have fun with.
// NOTE.2: This script is not here to teach you ANY JavaScript. It's aweful!
// NOTE.3: Main audience for this are people new to PhantomJS.

var sys = require("system"),
    page = require("webpage").create(),
    logResources = false,
    step1url = "http://en.wikipedia.org/wiki/DOM_events",
    step2url = "http://en.wikipedia.org/wiki/DOM_events#Event_flow";

if (sys.args.length > 1 && sys.args[1] === "-v") {
    logResources = true;
}

function printArgs() {
    var i, ilen;
    for (i = 0, ilen = arguments.length; i < ilen; ++i) {
        console.log("    arguments[" + i + "] = " + JSON.stringify(arguments[i]));
    }
    console.log("");
}

////////////////////////////////////////////////////////////////////////////////

page.onInitialized = function() {
    console.log("page.onInitialized");
    printArgs.apply(this, arguments);
};
page.onLoadStarted = function() {
    console.log("page.onLoadStarted");
    printArgs.apply(this, arguments);
};
page.onLoadFinished = function() {
    console.log("page.onLoadFinished");
    printArgs.apply(this, arguments);
};
page.onUrlChanged = function() {
    console.log("page.onUrlChanged");
    printArgs.apply(this, arguments);
};
page.onNavigationRequested = function() {
    console.log("page.onNavigationRequested");
    printArgs.apply(this, arguments);
};

if (logResources === true) {
    page.onResourceRequested = function() {
        console.log("page.onResourceRequested");
        printArgs.apply(this, arguments);
    };
    page.onResourceReceived = function() {
        console.log("page.onResourceReceived");
        printArgs.apply(this, arguments);
    };
}

page.onClosing = function() {
    console.log("page.onClosing");
    printArgs.apply(this, arguments);
};

// window.console.log(msg);
page.onConsoleMessage = function() {
    console.log("page.onConsoleMessage");
    printArgs.apply(this, arguments);
};

// window.alert(msg);
page.onAlert = function() {
    console.log("page.onAlert");
    printArgs.apply(this, arguments);
};
// var confirmed = window.confirm(msg);
page.onConfirm = function() {
    console.log("page.onConfirm");
    printArgs.apply(this, arguments);
};
// var user_value = window.prompt(msg, default_value);
page.onPrompt = function() {
    console.log("page.onPrompt");
    printArgs.apply(this, arguments);
};

////////////////////////////////////////////////////////////////////////////////

setTimeout(function() {
    console.log("");
    console.log("### STEP 1: Load '" + step1url + "'");
    page.open(step1url);
}, 0);

setTimeout(function() {
    console.log("");
    console.log("### STEP 2: Load '" + step2url + "' (load same URL plus FRAGMENT)");
    page.open(step2url);
}, 5000);

setTimeout(function() {
    console.log("");
    console.log("### STEP 3: Click on page internal link (aka FRAGMENT)");
    page.evaluate(function() {
        var ev = document.createEvent("MouseEvents");
        ev.initEvent("click", true, true);
        document.querySelector("a[href='#Event_object']").dispatchEvent(ev);
    });
}, 10000);

setTimeout(function() {
    console.log("");
    console.log("### STEP 4: Click on page external link");
    page.evaluate(function() {
        var ev = document.createEvent("MouseEvents");
        ev.initEvent("click", true, true);
        document.querySelector("a[title='JavaScript']").dispatchEvent(ev);
    });
}, 15000);

setTimeout(function() {
    console.log("");
    console.log("### STEP 5: Close page and shutdown (with a delay)");
    page.close();
    setTimeout(function(){
        phantom.exit();
    }, 100);
}, 20000);
