// Use container fluid
var containers = $(".container");
containers.removeClass("container");
containers.addClass("container-fluid");

WINDOW_CONTENTS = window.location.href.split('/')
var SELECTED_LANGUAGE = ''
var PACKAGE_REGEX = ''
var PACKAGE_REPLACEMENT = ''

ATTR1 = '[<span class="hljs-meta">System.ComponentModel.EditorBrowsable</span>]\n<'

// Navbar Hamburger
$(function () {
    $(".navbar-toggle").click(function () {
        $(this).toggleClass("change");
    })
})

// Select list to replace affix on small screens
$(function () {
    var navItems = $(".sideaffix .level1 > li");

    if (navItems.length == 0) {
        return;
    }

    var selector = $("<select/>");
    selector.addClass("form-control visible-sm visible-xs");
    var form = $("<form/>");
    form.append(selector);
    form.prependTo("article");

    selector.change(function () {
        window.location = $(this).find("option:selected").val();
    })

    function work(item, level) {
        var link = item.children('a');

        var text = link.text();

        for (var i = 0; i < level; ++i) {
            text = '&nbsp;&nbsp;' + text;
        }

        selector.append($('<option/>', {
            'value': link.attr('href'),
            'html': text
        }));

        var nested = item.children('ul');

        if (nested.length > 0) {
            nested.children('li').each(function () {
                work($(this), level + 1);
            });
        }
    }

    navItems.each(function () {
        work($(this), 0);
    });
})


$(function () {
    // Inject line breaks and spaces into the code sections
    $(".lang-csharp").each(function () {
        var text = $(this).html();
        text = text.replace(/, /g, ",</br>&#09;&#09");
        text = text.replace(ATTR1, '<');
        $(this).html(text);
    });

    // Add text to empty links
    $("p > a").each(function () {
        var link = $(this).attr('href')
        if ($(this).text() === "" && $(this).children().attr("src") === "") {
            $(this).html(link)
        }
    });
})

function httpGetAsync(targetUrl, callback) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function () {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
            callback(xmlHttp.responseText);
    }
    xmlHttp.open("GET", targetUrl, true); // true for asynchronous 
    xmlHttp.send(null);
}

function httpGetAsyncFallbackOnFail(targetUrl, successCallback, failureCallback) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function () {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                successCallback(xmlHttp.responseText);
            } else {
                failureCallback(xmlHttp.status)
            }
        }
    }
    xmlHttp.open("GET", targetUrl, true); // true for asynchronous 
    xmlHttp.send(null);
}

function populateOptions(selector, packageName) {
    var versionRequestUrl = "https://azuresdkdocs.blob.core.windows.net/$web/" + SELECTED_LANGUAGE + "/" + packageName + "/versioning/versions"

    httpGetAsync(versionRequestUrl, function (responseText) {
        var versionselector = document.createElement("select")
        var cv = WINDOW_CONTENTS[6]

        versionselector.className = 'navbar-version-select'
        if (responseText) {
            options = responseText.match(/[^\r\n]+/g)
            for (var i in options) {
                $(versionselector).append('<option value="' + options[i] + '">' + options[i] + '</option>')
            }
        }

        if(cv === 'latest')
        {
            $(versionselector).selectedIndex = 0
        }
        else {
            $(versionselector).val(cv);
        }
        
        $(selector).append(versionselector)

        $(versionselector).change(function () {
            targetVersion = $(this).val()
            url = WINDOW_CONTENTS.slice()
            url[6] = targetVersion
            var targetUrl = url.join('/')
            httpGetAsyncFallbackOnFail(targetUrl, (unused) => window.location.href = url.join('/'),
                (failureStatus) => window.location.href = getPackageUrl(SELECTED_LANGUAGE, packageName, targetVersion))
        });

    })
}

function httpGetLatestAsync(targetUrl, latestVersions, packageName) {
    httpGetAsync(targetUrl, function (responseText) {
        if (responseText) {
            version = responseText.match(/[^\r\n]+/g)
            $(latestVersions).append('<li><a href="' + getPackageUrl(SELECTED_LANGUAGE, packageName, version) + '" target="_blank">' + version + '</a></li>')
        }             
    })
}

function loadedOtherVersions(url, latestVersions, publishedVersions, selector, collapsible, packageName) {
    var hasAdded = function (currentVersion) {
        return $(publishedVersions).children('li').filter(function() {
            return $(this).text() === currentVersion
        }).length || $(latestVersions).children('li').filter(function() {
            return $(this).text() === currentVersion
        }).length
    }
    httpGetAsync(url, function (responseText) {
        if (responseText) {
            options = responseText.match(/[^\r\n]+/g)
            for (var i in options) {
                if (!hasAdded(options[i])) {
                    $(publishedVersions).append('<li><a href="' + getPackageUrl(SELECTED_LANGUAGE, packageName, options[i]) + '" target="_blank">' + options[i] + '</a></li>')
                }
            }
        }
        else {
            $(publishedVersions).append('<li>No discovered versions present in blob storage.</li>')
        }                
        $(selector).addClass("loaded")
        if ($(publishedVersions).children('li').length < 1) {
            $(collapsible).remove()
        }
    })
}

function populateIndexList(selector, packageName) {
    var url = "https://azuresdkdocs.blob.core.windows.net/$web/" + SELECTED_LANGUAGE + "/" + packageName + "/versioning/versions"
    var latestGAUrl = "https://azuresdkdocs.blob.core.windows.net/$web/" + SELECTED_LANGUAGE + "/" + packageName + "/versioning/latest-ga"
    var latestPreviewUrl = "https://azuresdkdocs.blob.core.windows.net/$web/" + SELECTED_LANGUAGE + "/" + packageName + "/versioning/latest-preview"
    var latestVersions = document.createElement("ul")
    httpGetLatestAsync(latestGAUrl, latestVersions, packageName)
    httpGetLatestAsync(latestPreviewUrl, latestVersions, packageName)
    var publishedVersions = $('<ul style="display: none;"></ul>')
    var collapsible = $('<div class="versionarrow">&nbsp;&nbsp;&nbsp;Other versions</div>')
    // Check whether it has display name tag.
    if ($(selector).next().is('h5')) {
        $(selector).next().after(latestVersions)
    } else {
        $(selector).after(latestVersions)
    }
    $(latestVersions).after(collapsible)
    $(collapsible).after(publishedVersions)

    // Add collapsible arrows on versioned docs.
    $(collapsible).on('click', function(event) {
        event.preventDefault();
        if (collapsible.hasClass('disable')) {
            return
        }
        $(this).toggleClass('down')
        if ($(this).hasClass('down')) {
            if (!$(selector).hasClass('loaded')){
                loadedOtherVersions(url, latestVersions, publishedVersions, selector, collapsible, packageName)
            }
            $(publishedVersions).show()
        } else {
            $(publishedVersions).hide()
        }
    });
}

function getPackageUrl(language, package, version) {
    return "https://azuresdkdocs.blob.core.windows.net/$web/" + language + "/" + package + "/" + version + "/index.html"
}

// Populate Versions
$(function () {
    if (WINDOW_CONTENTS.length < 7 && WINDOW_CONTENTS[WINDOW_CONTENTS.length - 1] != 'index.html') {
        console.log("Run PopulateList")

        $('h4').each(function () {
            var pkgName = $(this).text().replace(PACKAGE_REGEX, PACKAGE_REPLACEMENT)
            populateIndexList($(this), pkgName)
        })
    }

    if (WINDOW_CONTENTS.length > 7) {
        var pkgName = WINDOW_CONTENTS[5]
        populateOptions($('#navbar'), pkgName)
    }
})
