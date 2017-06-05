const docFolder = './documents/';
const fs = require('fs');
const marked = require('marked');
const highlight = require('highlight.js');
const renderer = new marked.Renderer();
const brodocDec = require('./markedDecorations.js');


marked.setOptions({
    renderer: renderer,
    gfm: true,
    tables: true,
    breaks: false,
    pedantic: false,
    sanitize: false,
    smartLists: true,
    smartypants: false,
    highlight: function (code, lang) {
        return highlight.highlightAuto(code).value;
    }
});
brodocDec.decorateMarked(renderer);

var config = require('./manifest');
var docs = config.docs;

var files = [];
var fileArray = [];
docs.forEach(file => {
    files.push(file.filename);
    fileArray.push(file);
});

var bodyContent = '';
var navIds = brodocDec.navIds;
var codeTabs = brodocDec.codeTabs;


// const lexer = new marked.Lexer();
// lexer.rules.bdoc = /^(\/{4} )(\w+).*$/;

var path = docFolder;
var fIndex = 0;
var rIndex = 0;
var fileObj = {toc: [], content: [], tabs: []};
fileArray.forEach((file, index) => {
    fs.readFile(path + file.filename, 'utf8', (err, data) => {
        rIndex++;
        file.content = data;

        if (rIndex >= files.length) {
            // do the things
            parseFileContent(fileArray);
            var navData = generateNavItems(navIds);
            var navContent = navData.content;
            var navDataArray = navData.navDataArray;
            var codeTabContent = generateCodeTabItems(codeTabs);
            var bodyContent = flattenContent(parsedContentArray);
            generateDoc(navContent, bodyContent, codeTabContent);
            generateNavJson(navDataArray);
        }
    });
});

function flattenContent(content) {
    var flattenedContent = content.reduce(function(accum, val) {
        return accum + val;
    });
    return flattenedContent;
}

var parsedContentArray = [];
function parseFileContent(files) {
    files.forEach((file, index) => {
        parsedContentArray[index] = parseDoc(file.content);
    });
}
function parseDoc(doc) {
    return marked(doc, { renderer: renderer });
}

function generateNavItems(navObjs) {
    var reversedNavs = navObjs.reverse();
    var currentNestArray = [];
    var currentStrongArray = [];
    var flattenedNest = '';
    var nestedNavArray = []; // Array containing generated html menu items - is flattened into a string.
    var navArrayInvert = []; // Deals with data layer of navigation;
    var navSectionArray = [];
    var navStrongSectionArray = [];
    var navSectionArrayClone;
    var flatNavArrayInvert = [];
    reversedNavs.forEach(obj => {
        flatNavArrayInvert.push(obj.id);
        var strong = (obj.id.indexOf('-strong-') !== -1);
        if (obj.level !== 1) {
            if (strong && currentNestArray.length !== 0) {
                flattenedNest = flattenContent(currentNestArray.reverse());
                currentStrongArray.push(generateNestedNav(obj, flattenedNest));
                currentNestArray.length = 0;

                navSectionArrayClone = Object.assign([], navSectionArray);
                navStrongSectionArray.push({section: obj.id, subsections: navSectionArrayClone});
                navSectionArray.length = 0;
            } else {
                currentNestArray.push(generateNav(obj));
                navSectionArray.push({section: obj.id});
            }
        } else if (obj.level === 1) {
            if (currentStrongArray.length !== 0) {
                currentNestArray.forEach(obj => {
                    currentStrongArray.push(obj);
                });
                flattenedNest = flattenContent(currentStrongArray.reverse());
            } else if (currentNestArray.length !== 0) {
                flattenedNest = flattenContent(currentNestArray.reverse());
            }
            nestedNavArray.push(generateNestedNav(obj, flattenedNest));
            currentNestArray.length = 0;
            currentStrongArray.length = 0;
            flattenedNest = '';

            navSectionArray.forEach(obj => {
                navStrongSectionArray.push(obj);
            });
            navSectionArrayClone = Object.assign([], navStrongSectionArray);
            navStrongSectionArray.length = 0;
            navArrayInvert.push({section: obj.id, subsections: navSectionArrayClone});
            navSectionArray.length = 0;
        }
    });
    
    var navContent = flattenContent(nestedNavArray.reverse());
    return {content: navContent, navDataArray: {toc: navArrayInvert, flatToc: flatNavArrayInvert}};
}

function generateNav(obj) {
    var classString = 'nav-level-' + obj.level;
    var isStrong = obj.id.indexOf('-strong-') !== -1;
    if (isStrong) {
        classString += ' strong-nav';
    }
    return '<li class="' + classString + '">' + '<a href="#' + obj.id + '" class="nav-item">' + obj.text + '</a></li>';
}

function generateNestedNav(parent, nest) {
    var nestContent = '';
    if (nest.length > 0) {
        nestContent = nest ? '<ul id="' + parent.id + '-nav" style="display: none;">' + nest + '</ul>' : '';
    }
    return '<ul>' + generateNav(parent) + nestContent + '</ul>';
}

function generateNavJson(data) {
    var navJson = JSON.stringify(data);
    navScript = `(function(){navData = ${navJson}})();`;
    fs.writeFile('./navData.js', navScript, function(err) {
        if (err) {
            return console.log(err);
        }
        console.log("navData.js saved!");
    });
}

function generateCodeTabItems(tabs) {
    var codeTabList = '';
    tabs.forEach(tab => {
        codeTabList += generateCodeTab(tab);
    });
    return codeTabList;
}

function generateCodeTab(tab) {
    return '<li class="code-tab" id="' + tab + '">' + tab + '</li>';
}

function generateDoc(navContent, bodyContent, codeTabContent) {
    var doc = 
`<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>${config.title}</title>
<link rel="shortcut icon" href="favicon.ico" type="image/vnd.microsoft.icon">
<!-- Latest compiled and minified CSS -->
<link rel="stylesheet" href="node_modules/bootstrap/dist/css/bootstrap.min.css">
<link rel="stylesheet" href="node_modules/font-awesome/css/font-awesome.min.css" type="text/css">
<link rel="stylesheet" href="node_modules/highlight.js/styles/default.css" type="text/css">
<link rel="stylesheet" href="stylesheet.css" type="text/css">
</head>
<body>
<div id="sidebar-wrapper" class="side-nav side-bar-nav">${navContent}<br/><div class="copyright">${config.copyright}</div></div>
<div id="wrapper">
<div id="code-tabs-wrapper" class="code-tabs"><ul class="code-tab-list">${codeTabContent}</ul></div>
<div id="page-content-wrapper" class="body-content container-fluid">${bodyContent}</div>
</div>
<script src="node_modules/jquery/dist/jquery.min.js"></script>
<script src="node_modules/jquery.scrollto/jquery.scrollTo.min.js"></script>
<script src="navData.js"></script>
<script src="scroll.js"></script>
<!--<script src="actions.js"></script>-->
<script src="tabvisibility.js"></script>
</body>
</html>`;
    fs.writeFile('./index.html', doc, function (err) {
        if (err) {
            return console.log(err);
        }
        console.log("index.html saved!");
    });
}