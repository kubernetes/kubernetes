//var data = [4, 8, 15, 16, 23, 42];

function defaults(){

    Chart.defaults.global.animation = false;

}

function f(data2) {

    defaults();

    // Get context with jQuery - using jQuery's .get() method.
    var ctx = $("#myChart").get(0).getContext("2d");
    ctx.width  = $(window).width()*1.5;
    ctx.width  = $(window).height *.5;

    // This will get the first returned node in the jQuery collection.
    var myNewChart = new Chart(ctx);

    var data = {
        labels: Array.apply(null, Array(data2.length)).map(function (_, i) {return i;}),
        datasets: [
            {
                label: "My First dataset",
                fillColor: "rgba(220,220,220,0.2)",
                strokeColor: "rgba(220,220,220,1)",
                pointColor: "rgba(220,220,220,1)",
                pointStrokeColor: "#fff",
                pointHighlightFill: "#fff",
                pointHighlightStroke: "rgba(220,220,220,1)",
                data: data2
            }
        ]
    };

    var myLineChart = new Chart(ctx).Line(data);
}

