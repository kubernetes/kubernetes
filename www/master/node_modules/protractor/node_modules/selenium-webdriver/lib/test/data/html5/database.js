var database={};
database.db={};

database.onError = function(tx, e) {
  var log = document.createElement('div');
  log.setAttribute('name','error');
  log.setAttribute('style','background-color:red');
  log.innerText = e.message;
  document.getElementById('logs').appendChild(log);
}

database.onSuccess = function(tx, r) {
  if (r.rows.length) {
    var ol;
    for (var i = 0; i < r.rows.length; i++) {
      ol = document.createElement('ol');
      ol.innerHTML = r.rows.item(i).ID + ": " + r.rows.item(i).docname + " (" + r.rows.item(i).created + ")";
      document.getElementById('logs').appendChild(ol);
    }

  }
}

database.open=function(){
  database.db=openDatabase('HTML5', '1.0', 'Offline document storage', 100*1024);
}

database.create=function(){
  database.db.transaction(function(tx) {
    tx.executeSql("CREATE TABLE IF NOT EXISTS docs(ID INTEGER PRIMARY KEY ASC, docname TEXT, created TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
      [],
      database.onSuccess,
      database.onError);
  });}

database.add = function(message) {
  database.db.transaction(function(tx){
    tx.executeSql("INSERT INTO docs(docname) VALUES (?)",
        [message], database.onSuccess, database.onError);
    });
}

database.selectAll = function() {
  database.db.transaction(function(tx) {
    tx.executeSql("SELECT * FROM docs", [], database.onSuccess,
        database.onError);
 });
}

database.onDeleteAllSuccess = function(tx, r) {
  var doc = document.documentElement;
  var db_completed = document.createElement("div");
  db_completed.setAttribute("id", "db_completed");
  db_completed.innerText = "db operation completed";
  doc.appendChild(db_completed);
}

database.deleteAll = function() {
  database.db.transaction(function(tx) {
    tx.executeSql("delete from docs", [], database.onDeleteAllSuccess,
        database.onError);
  });
}

var log = document.createElement('div');
log.setAttribute('name','notice');
log.setAttribute('style','background-color:yellow');
log.innerText = typeof window.openDatabase == "function" ? "Web Database is supported." : "Web Database is not supported.";
document.getElementById('logs').appendChild(log);

try {
  database.open();
  database.create();
  database.add('Doc 1');
  database.add('Doc 2');
  database.selectAll();
  database.deleteAll();
} catch(error) {
  var log = document.createElement('div');
  log.setAttribute('name','critical');
  log.setAttribute('style','background-color:pink');
  log.innerText =  error;
  document.getElementById('logs').appendChild(log);
}
