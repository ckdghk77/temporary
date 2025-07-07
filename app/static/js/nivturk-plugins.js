// Pass message from jsPsych to NivTurk
function pass_message(msg) {

  $.ajax({
    url: "/experiment",
    method: 'POST',
    data: JSON.stringify(msg),
    contentType: "application/json; charset=utf-8",
  }).done(function(data, textStatus, jqXHR) {
    // do nothing on success
  }).fail(function(error) {
    console.log(error);
  });

}

// Save an incomplete dataset.
function incomplete_save() {

  $.ajax({
    url: "/incomplete_save",
    method: 'POST',
    data: JSON.stringify(jsPsych.data.get().json()),
    contentType: "application/json; charset=utf-8",
  }).done(function(data, textStatus, jqXHR) {
    // do nothing
  }).fail(function(error) {
    // do nothing
  });

}

// Successful completion of experiment: redirect with completion code.
function redirect_success(workerId, assignmentId, hitId, a, tp_a, b, tp_b, c, tp_c) {

  // Concatenate metadata into complete URL (returned on success).
  var url = "/complete?workerId=" + workerId + "&assignmentId=" + assignmentId + "&hitId=" + hitId + "&a=" + a + "&tp_a=" + tp_a + "&b=" + b + "&tp_b=" + tp_b + "&c=" + c + "&tp_c=" + tp_c;

  $.ajax({
    url: "/redirect_success",
    method: 'POST',
    data: JSON.stringify(jsPsych.data.get().json()),
    contentType: "application/json; charset=utf-8",
  }).done(function(data, textStatus, jqXHR) {
    window.location.replace(url);
  }).fail(function(error) {
    window.location.replace(url);
  });

}

// Unsuccessful completion of experiment: redirect with decoy code.
function redirect_reject(error) {

  // Concatenate metadata into complete URL (returned on reject).
  var url = "/error/" + error;

  $.ajax({
    url: "/redirect_reject",
    method: 'POST',
    data: JSON.stringify(jsPsych.data.get().json()),
    contentType: "application/json; charset=utf-8",
  }).done(function(data, textStatus, jqXHR) {
    window.location.replace(url);
  }).fail(function(error) {
    window.location.replace(url);
  });
}

// Unsuccessful completion of experiment: redirect to error page.
function redirect_error(error) {

  // error is the error number to redirect to.
  var url = "/error/" + error;

  $.ajax({
    url: "/redirect_error",
    method: 'POST',
    data: JSON.stringify(jsPsych.data.get().json()),
    contentType: "application/json; charset=utf-8",
  }).done(function(data, textStatus, jqXHR) {
    window.location.replace(url);
  }).fail(function(error) {
    window.location.replace(url);
  });

}

// After session end, post all entire data for training model
// program on "/send_data" is defined in experiment.py
function send_data() {
  
  $.ajax({
    url: "/send_data",
    method: 'POST',
    data: JSON.stringify(jsPsych.data.get().json()),
    contentType: "application/json; charset=utf-8",
  }).done(function(data, textStatus, jqXHR) { 

      // If communication with server succeed
      // server returns json type or text type 
      // return file should have 3 keys (cue_idxes, cue_allocation, reward_allocation)
      
      ///// check communication
      console.log("success") 
      
      // console.log(data)

      ///// this function replace alloc_all()
      //used_cue_idxes = data.cue_idxes;
      //main_cue_allocation = data.cue_allocation;
      //main_reward_allocation = data.reward_allocation;

  }).fail(function(error) {
      // alert("fail")
    // do nothing
  });

}

// After session end, get progress data of training model
// program on "/count_epoch" is defined in experiment.py
// (send_data() + count_epoch()) only works well at single process environment 
function count_epoch() {
  
  $.ajax({
    url: "/train_iter",
    method: 'GET',
    dataType: 'text',
    // data: JSON.stringify(jsPsych.data.get().json()), // do not send data
    //contentType: "application/json; charset=utf-8",
  }).done(function(data, textStatus, jqXHR) { 

      // If communication with server succeed

      //console.log(performance.now());
      loaded_model = parseFloat(data);
      
      ///// check communication
      // console.log("success") 
      // console.log(data)

  }).fail(function(error) {
    // do nothing
  });


}

// After session end, get current epoch/total epoch when each epoch is finished
// train_iter() works well at multi process environment 
// program on "/train_iter" is defined in experiment.py
function train_iter() {
  
  $.ajax({
    url: "/train_iter",
    method: 'GET',
    dataType: 'text',
    async: false,
    // data: JSON.stringify(jsPsych.data.get().json()), // do not send data
    //contentType: "application/json; charset=utf-8",
  }).done(function(data, textStatus, jqXHR) { 

      // If communication with server succeed

      //console.log(performance.now());
      loaded_model = parseFloat(data);
      
      ///// check communication
      // console.log("success") 
      // console.log(data)

  }).fail(function(error) {
    // do nothing
  });


}

