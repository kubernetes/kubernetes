// List movies from kids-in-mind.com

var cbfunc = function (data) {
    globaldata= data;
    var list = data.query.results.movie;
    list.forEach(function (item) {
        console.log(item.title + ' [' + item.rating.MPAA.content + ']');
    });
    phantom.exit();
};

var el = document.createElement('script');
el.src = 'http://query.yahooapis.com/v1/public/yql?q=select%20*%20from%20movies.kids-in-mind&format=json&env=store%3A%2F%2Fdatatables.org%2Falltableswithkeys&callback=cbfunc';
document.body.appendChild(el);
