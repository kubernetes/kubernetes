// allow the user to store recent queries for quick retrieval
var recentQueries = [];
var queryPointer = null;

// keep track of the databases that exist on the server
var availableDatabases = [];
var currentlySelectedDatabase = null;

// connection settings for the server, with sensible defaults
var connectionSettings = {
    hostname: (window.location.hostname ? window.location.hostname: "localhost"),
    port: "8086",
    username: "",
    password: "",
    ssl: ('https:' == window.location.protocol ? true : false)
}

var connectionString = function() {
    var protocol = (connectionSettings.ssl ? "https" : "http");
    var host = connectionSettings.hostname + ":" + connectionSettings.port;

    if (connectionSettings.username !== "") {
        $.ajaxSetup({
            headers: {
                'Authorization': "Basic " + btoa(connectionSettings.username + ":" + connectionSettings.password)
            }
        });
    }

    return protocol + "://" + host;
}

var getSeriesFromJSON = function(data) {
    var results = data.results[0].series;
    return results;
}

// gets settings from the browser's localStorage and sets defaults if they aren't found
var loadSettings = function() {
    var cs = localStorage.getItem("connectionSettings");

    if (cs != null) { connectionSettings = JSON.parse(cs); }

    document.getElementById('hostname').value = connectionSettings.hostname;
    document.getElementById('port').value = connectionSettings.port;
    document.getElementById('username').value = connectionSettings.username;
    document.getElementById('password').value = connectionSettings.password;
    document.getElementById('ssl').checked = connectionSettings.ssl;

    getDatabases();
}

var updateSettings = function() {
    var hostname = document.getElementById('hostname').value;
    var port = document.getElementById('port').value;
    var username = document.getElementById('username').value;
    var password = document.getElementById('password').value;
    var ssl = document.getElementById('ssl').checked;

    if (hostname == "") { hostname = "localhost"; }

    if (port == "") { port = "8086"; }

    connectionSettings.hostname = hostname;
    connectionSettings.port     = port;
    connectionSettings.username = username;
    connectionSettings.password = password;
    connectionSettings.ssl      = ssl;

    localStorage.setItem("connectionSettings", JSON.stringify(connectionSettings));

    getDatabases();
}

var showSettings = function() {
    $("#settings").show();
    $("input#query").prop('disabled', true);
}

var hideSettings = function() {
    $("#settings").hide();
    $("input#query").prop('disabled', false);
}

// hide errors within the Write Data modal
var hideModalError = function() {
    $("div#modal-error").empty().hide();
}

// show errors within the Write Data modal
var showModalError = function(message) {
    hideModalSuccess();

    $("div#modal-error").html("<p>" + message + "</p>").show();
}

// hide success messages within the Write Data modal
var hideModalSuccess = function() {
    $("div#modal-success").empty().hide();
}

// show success messages within the Write Data modal
var showModalSuccess = function(message) {
    hideModalError();

    $("div#modal-success").html("<p>" + message + "</p>").show();
}

// hide errors from queries
var hideQueryError = function() {
    $("div#query-error").empty().hide();
}

// show errors from queries
var showQueryError = function(message) {
    hideQuerySuccess();

    $("div#query-error").html("<p>" + message + "</p>").show();
}

// hide success messages from queries
var hideQuerySuccess = function() {
    $("div#query-success").empty().hide();
}

// show success messages from queries
var showQuerySuccess = function(message) {
    hideQueryError();

    $("div#query-success").html("<p>" + message + "</p>").show();
}

// hide warning from database lookup
var hideDatabaseWarning = function() {
    $("div#database-warning").empty().hide();
}

// show warning from database lookup
var showDatabaseWarning = function(message) {
    $("div#database-warning").html("<p>" + message + "</p>").show();
}

// clear out the results table
var clearResults = function() {
    $("div#table").empty();
}

// handle submissions of the query bar
var handleSubmit = function(e) {
    var queryElement = document.getElementById('query');
    var q = queryElement.value;

    clearResults();

    if (q == "") { return };

    var query = $.get(connectionString() + "/query", {q: q, db: currentlySelectedDatabase}, function() {
        hideQueryError();
        hideQuerySuccess();
    });

    recentQueries.push(q);
    queryPointer = recentQueries.length - 1;

    query.fail(handleRequestError);

    query.done(function (data) {
        var firstRow = data.results[0];
        if (firstRow.error) {
            showQueryError("Server returned error: " + firstRow.error);
            return
        }

        var series = getSeriesFromJSON(data);

        if (series == null) {
            showQuerySuccess("Success! (no results to display)");
            getDatabases();
            return
        }

        var values = series[0].values;

        if ((values == null) || (values.length == 0)) {
            showQueryError("Query returned no results!");
        } else {
            availableDatabases = values.map(function(value) {
                return value[0];
            });

            hideDatabaseWarning();
            React.render(
              React.createElement(DataTable, {series: series}),
              document.getElementById('table')
            );
        }
    });

    if (e != null) { e.preventDefault(); }
    return false;
};

var handleRequestError = function(e) {
    var errorText = e.status + " " + e.statusText;
    showDatabaseWarning("Unable to fetch list of databases.");

    if ("responseText" in e) {
        try { errorText = "Server returned error: " + JSON.parse(e.responseText).error; } catch(e) {}
    }

    if (e.status == 400) {
        hideSettings();
    } else if (e.status == 401) {
        if (errorText.indexOf("error authorizing query") > -1) {
            hideSettings();
            $("input#query").val("CREATE USER <username> WITH PASSWORD '<password>' WITH ALL PRIVILEGES").focus();
        } else {
            showSettings();
            $("input#username").focus();
        }
    } else {
        showSettings();
        $("input#hostname").focus();
        showDatabaseWarning("Hint: the InfluxDB API runs on port 8086 by default");
        errorText = e.status + " " + e.statusText + " - Could not connect to " + connectionString();
    }
    showQueryError(errorText);
};

var handleKeypress = function(e) {
    var queryElement = document.getElementById('query');

    // key press == enter
    if (e.keyCode == 13) {
        e.preventDefault();
        handleSubmit();
        return false;
    }

    // if we don't have any recent queries, ignore the arrow keys
    if (recentQueries.length == 0 ) { return }

    // key press == up arrow
    if (e.keyCode == 38) {
        // TODO: stash the current query, if there is one?
        if (queryPointer == recentQueries.length - 1) {
            // this is buggy.
            //recentQueries.push(queryElement.value);
            //queryPointer = recentQueries.length - 1;
        }

        if (queryPointer != null && queryPointer > 0) {
            queryPointer -= 1;
            queryElement.value = recentQueries[queryPointer];
        }
    }

    // key press == down arrow
    if (e.keyCode == 40) {
        if (queryPointer != null && queryPointer < recentQueries.length - 1) {
            queryPointer += 1;
            queryElement.value = recentQueries[queryPointer];
        }
    }
};

var QueryError = React.createClass({
    render: function() {
        return React.createElement("div", {className: "alert alert-danger"}, this.props.message)
    }
});

var stringifyTags = function(tags) {
    var tagStrings = [];

    for(var index in tags) {
        tagStrings.push(index + ":" + tags[index]);
    }

    return tagStrings.join(", ");
}

var DataTable = React.createClass({
  render: function() {
    var tables = this.props.series.map(function(series) {
        return React.createElement("div", null,
            React.createElement("h1", null, series.name),
            React.createElement("h2", null, stringifyTags(series.tags)),
            React.createElement("table", {className: "table"},
                React.createElement(TableHeader, {data: series.columns}),
                React.createElement(TableBody, {data: series})
            )
        );
    });

    return React.createElement("div", null, tables);
  }
});

var TableHeader = React.createClass({
    render: function() {
        var headers = this.props.data.map(function(column) {
            return React.createElement("th", null, column);
        });

        return React.createElement("tr", null, headers);
    }
});

var TableBody = React.createClass({
    render: function() {
        if (this.props.data.values) {
            var tableRows = this.props.data.values.map(function (row) {
                return React.createElement(TableRow, {data: row});
            });
        }

        return React.createElement("tbody", null, tableRows);
    }
});

var TableRow = React.createClass({
    render: function() {
        var tableData = this.props.data.map(function (data, index) {
            if (index == 0) {
                return React.createElement("td", {className: "timestamp"}, null, data);
            } else {
                return React.createElement("td", null, pretty(data));
            }
        });

        return React.createElement("tr", null, tableData);
    }
});

var pretty = function(val) {
    if (typeof val == 'string') {
        return "\"" + val + "\"";
    } else if (typeof val == 'boolean' ){
        return val.toString();
    } else {
        return val;
    }
}

var chooseDatabase = function (databaseName) {
    currentlySelectedDatabase = databaseName;
    document.getElementById("content-current-database").innerHTML = currentlySelectedDatabase;
}

var getDatabases = function () {
    var q = "SHOW DATABASES";
    var query = $.get(connectionString() + "/query", {q: q, db: currentlySelectedDatabase});

    query.fail(handleRequestError);

    query.done(function (data) {
        hideSettings();
        hideDatabaseWarning();

        var firstRow = data.results[0];
        if (firstRow.error) {
            showDatabaseWarning(firstRow.error);
            return;
        }

        var series = getSeriesFromJSON(data);
        var values = series[0].values;

        if ((values == null) || (values.length == 0)) {
            availableDatabases = [];
            updateDatabaseList();

            showDatabaseWarning("No databases found.")
        } else {
            availableDatabases = values.map(function(value) {
                return value[0];
            }).sort();

            if (currentlySelectedDatabase == null) {
                chooseDatabase(availableDatabases[0]);
            } else if (availableDatabases.indexOf(currentlySelectedDatabase) == -1) {
                chooseDatabase(availableDatabases[0]);
            }
            updateDatabaseList();
        }
    });
}

var updateDatabaseList = function() {
    var databaseList = $("ul#content-database-list");

    databaseList.empty();
    availableDatabases.forEach(function(database) {
        var li = $("<li><a href=\"#\">" + database + "</a></li>");
        databaseList.append(li);
    });

    if (availableDatabases.length == 0) {
        document.getElementById("content-current-database").innerHTML = "&hellip;";
    }
}

// when the page is ready, start everything up
$(document).ready(function () {
    loadSettings();

    // bind to the settings cog in the navbar
    $("#action-settings").click(function (e) {
        $("#settings").toggle();
    });

    // bind to the save button in the settings form
    $("#form-settings").submit(function (e) {
        updateSettings();
    });

    // bind to the items in the query template dropdown
    $("ul#action-template label").click(function (e) {
        var el = $(e.target);
        $("input#query").val(el.data("query")).focus();
    });

    $("ul#content-database-list").on("click", function(e) {
        if (e.target.tagName != "A") { return; }

        chooseDatabase(e.target.innerHTML);
        e.preventDefault();
    })

    // load the Write Data modal
    $("button#action-send").click(function (e) {
        var data = $("textarea#content-data").val();

        var startTime = new Date().getTime();
        var write = $.post(connectionString() + "/write?db=" + currentlySelectedDatabase, data, function() {
        });

        write.fail(function (e) {
            if (e.status == 400) {
                showModalError("Failed to write: " + e.responseText)
            }
            else {
                showModalError("Failed to contact server: " + e.statusText)
            }
        });

        write.done(function (data) {
            var endTime = new Date().getTime();
            var elapsed = endTime - startTime;
            showModalSuccess("Write succeeded. (" + elapsed + "ms)");
        });

    });

    // handle submit actions on the query bar
    var form = document.getElementById('query-form');
    form.addEventListener("submit", handleSubmit);

    // handle keypresses on the query bar so we can get arrow keys and enter
    var query = document.getElementById('query');
    query.addEventListener("keydown", handleKeypress);

    // make sure we start out with the query bar in focus
    document.getElementById('query').focus();
})
