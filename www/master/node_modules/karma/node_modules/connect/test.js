
var conn = require('./');
var app = conn();

app.use(conn.logger('dev'));

app.listen(3000);
