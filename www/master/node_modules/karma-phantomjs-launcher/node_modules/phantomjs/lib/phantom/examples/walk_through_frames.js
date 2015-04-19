var p = require("webpage").create();

function pageTitle(page) {
    return page.evaluate(function(){
        return window.document.title;
    });
}

function setPageTitle(page, newTitle) {
    page.evaluate(function(newTitle){
        window.document.title = newTitle;
    }, newTitle);
}

p.open("../test/webpage-spec-frames/index.html", function(status) {
    console.log("pageTitle(): " + pageTitle(p));
    console.log("currentFrameName(): "+p.currentFrameName());
    console.log("childFramesCount(): "+p.childFramesCount());
    console.log("childFramesName(): "+p.childFramesName());
    console.log("setPageTitle(CURRENT TITLE+'-visited')"); setPageTitle(p, pageTitle(p) + "-visited");
    console.log("");

    console.log("p.switchToChildFrame(\"frame1\"): "+p.switchToChildFrame("frame1"));
    console.log("pageTitle(): " + pageTitle(p));
    console.log("currentFrameName(): "+p.currentFrameName());
    console.log("childFramesCount(): "+p.childFramesCount());
    console.log("childFramesName(): "+p.childFramesName());
    console.log("setPageTitle(CURRENT TITLE+'-visited')"); setPageTitle(p, pageTitle(p) + "-visited");
    console.log("");

    console.log("p.switchToChildFrame(\"frame1-2\"): "+p.switchToChildFrame("frame1-2"));
    console.log("pageTitle(): " + pageTitle(p));
    console.log("currentFrameName(): "+p.currentFrameName());
    console.log("childFramesCount(): "+p.childFramesCount());
    console.log("childFramesName(): "+p.childFramesName());
    console.log("setPageTitle(CURRENT TITLE+'-visited')"); setPageTitle(p, pageTitle(p) + "-visited");
    console.log("");

    console.log("p.switchToParentFrame(): "+p.switchToParentFrame());
    console.log("pageTitle(): " + pageTitle(p));
    console.log("currentFrameName(): "+p.currentFrameName());
    console.log("childFramesCount(): "+p.childFramesCount());
    console.log("childFramesName(): "+p.childFramesName());
    console.log("setPageTitle(CURRENT TITLE+'-visited')"); setPageTitle(p, pageTitle(p) + "-visited");
    console.log("");

    console.log("p.switchToChildFrame(0): "+p.switchToChildFrame(0));
    console.log("pageTitle(): " + pageTitle(p));
    console.log("currentFrameName(): "+p.currentFrameName());
    console.log("childFramesCount(): "+p.childFramesCount());
    console.log("childFramesName(): "+p.childFramesName());
    console.log("setPageTitle(CURRENT TITLE+'-visited')"); setPageTitle(p, pageTitle(p) + "-visited");
    console.log("");

    console.log("p.switchToMainFrame()"); p.switchToMainFrame();
    console.log("pageTitle(): " + pageTitle(p));
    console.log("currentFrameName(): "+p.currentFrameName());
    console.log("childFramesCount(): "+p.childFramesCount());
    console.log("childFramesName(): "+p.childFramesName());
    console.log("setPageTitle(CURRENT TITLE+'-visited')"); setPageTitle(p, pageTitle(p) + "-visited");
    console.log("");

    console.log("p.switchToChildFrame(\"frame2\"): "+p.switchToChildFrame("frame2"));
    console.log("pageTitle(): " + pageTitle(p));
    console.log("currentFrameName(): "+p.currentFrameName());
    console.log("childFramesCount(): "+p.childFramesCount());
    console.log("childFramesName(): "+p.childFramesName());
    console.log("setPageTitle(CURRENT TITLE+'-visited')"); setPageTitle(p, pageTitle(p) + "-visited");
    console.log("");

    phantom.exit();
});

