var $code   =  $('.code')
  , $config =  $('.config')
  , $result =  $('.result')
  ;

function go () {
  var config;
  try {
     config = JSON.parse($config.val());
  } catch (e) {
    $result.val('In "Redeyed Config": ' + e.toString());
    return;
  }

  try {
    var code = $code.val()
      , result = redeyed(code, config);

    $result.val(result.code);
  } catch (e) {
    $result.val('In "Original Code": ' + e.toString());
  }
}

$code.val(window.redeyed.toString());

$config.val(JSON.stringify(window.sampleConfig, false, 2));

$('.go').click(go);

go();


