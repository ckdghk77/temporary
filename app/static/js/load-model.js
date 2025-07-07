/**
 * jspsych-preload
 * documentation: docs.jspsych.org
 **/

jsPsych.plugins['load model'] = (function() {

    var plugin = {};

    plugin.info = {
      name: 'load model',
      description: '',
      parameters: {
        auto_preload: {
          type: jsPsych.plugins.parameterType.BOOL,
          default: false,
          description: 'Whether or not to automatically preload any media files based on the timeline passed to jsPsych.init.'
        },
        trials: {
          type: jsPsych.plugins.parameterType.TIMELINE,
          default: [],
          description: 'Array with a timeline of trials to automatically preload. If one or more trial objects is provided, '+
          'then the plugin will attempt to preload the media files used in the trial(s).'
        },
        session_num: {
          type: jsPsych.plugins.parameterType.INT,
          default: null,
          description: 'HTML-formatted message to be shown above the progress bar while the files are loading.'
        },
        message: {
          type: jsPsych.plugins.parameterType.HTML_STRING,
          default: null,
          description: 'HTML-formatted message to be shown above the progress bar while the files are loading.'
        },
        driving: {
          type: jsPsych.plugins.parameterType.INT,
          default: 0,
          description: 'driving on/off'
        },
        
        continue_after_error: {
          type: jsPsych.plugins.parameterType.BOOL,
          default: false,
          description: 'Whether or not to continue with the experiment if a loading error occurs. If false, then if a loading error occurs, '+
          'the error_message will be shown on the page and the trial will not end. If true, then if if a loading error occurs, the trial will end '+
          'and preloading failure will be logged in the trial data.'
        },
        error_message: {
          type: jsPsych.plugins.parameterType.HTML_STRING,
          default: 'The experiment failed to load.',
          description: 'Error message to show on the page in case of any loading errors. This parameter is only relevant when continue_after_error is false.'
        },
        show_detailed_errors: {
          type: jsPsych.plugins.parameterType.BOOL,
          default: false,
          description: 'Whether or not to show a detailed error message on the page. If true, then detailed error messages will be shown on the '+
          'page for all files that failed to load, along with the general error_message. This parameter is only relevant when continue_after_error is false.'
        },
        max_load_time: {
          type: jsPsych.plugins.parameterType.INT,
          default: null,
          description: 'The maximum amount of time that the plugin should wait before stopping the preload and either ending the trial '+
          '(if continue_after_error is true) or stopping the experiment with an error message (if continue_after_error is false). '+
          'If null, the plugin will wait indefintely for the files to load.'
        },
        on_error: {
          type: jsPsych.plugins.parameterType.FUNCTION,
          default: null,
          description: 'Function to be called after a file fails to load. The function takes the file name as its only argument.'
        },
        on_success: {
          type: jsPsych.plugins.parameterType.FUNCTION,
          default: null,
          description: 'Function to be called after a file loads successfully. The function takes the file name as its only argument.'
        }
      }
    }

    plugin.trial = function(display_element, trial) {

      var success = null;
      var timeout = false;
      var startTime = performance.now();
      var endTime = null;

      // render display of message and progress bar

      var html = '';
      var sub_html = '';
      loaded_model = 0;

      html += '<p>End of session #'+ (trial.session_num) +'</p>'

      if(trial.message !== null){
        html += trial.message;
      }

      sub_html = html;

      html += `
          <div id='jspsych-loading-progress-bar-container' style='height: 10px; width: 300px; background-color: #ddd; margin: auto;'>
            <div id='jspsych-loading-progress-bar' style='height: 10px; width: 0%; background-color: #777;'></div>
          </div>`;

      sub_html +=  `
          <div id='jspsych-loading-progress-bar-container' style='height: 10px; width: 300px; background-color: #ddd; margin: auto;'>
            <div id='jspsych-loading-progress-bar' style='height: 10px; width: 100%; background-color: #777;'></div>
          </div>`;

      html += "<p style='margin-top:50px;'>&nbsp&nbsp</p>" // space for show_session_end msg

      display_element.innerHTML = html;

      if(trial.max_load_time !== null){
        jsPsych.pluginAPI.setTimeout(on_timeout, trial.max_load_time);
      }

      // jsPsych.data.get().localSave('json','test.json')
      // get loading progress data
      incomplete_save()
      model_loading();

      // function to get model progress
      function model_loading() {
        jsPsych.pluginAPI.setTimeout(function(){

          if(trial.driving==1){
            train_iter()
          }  
          else{
            $.ajax({
              type: "GET",
              url: `/dummy_iter`,
              async: false,
              success: function(response){
                loaded_model = JSON.parse(response).progress
              }
            }
            )
          } 

          //console.log(loaded_model);

          // update_loading_progress_bar
          var percent_loaded = (loaded_model)*100;
          var preload_progress_bar = jsPsych.getDisplayElement().querySelector('#jspsych-loading-progress-bar');
          if (preload_progress_bar !== null) {
            preload_progress_bar.style.width = percent_loaded+"%";
          }

          if (loaded_model == 1) {
            success = true;
            endTime = performance.now();
            jsPsych.pluginAPI.setTimeout(function(){
              end_trial();
            },500)
            return;
          } else if (loaded_model < 1) {
            model_loading();
            return;
          }
        }
        ,500) // time interval to next training
      }
      
      function end_trial(){
        // clear timeout again when end_trial is called, to handle race condition with max_load_time
        jsPsych.pluginAPI.clearAllTimeouts();

        var trial_data = {
          success: success,
          timeout: timeout,
          start: startTime,
          end: endTime,
        };
        // clear the display
        session_end_msg = sub_html;
        display_element.innerHTML = '';
        jsPsych.finishTrial(trial_data);
      }
    };

    return plugin;
  })();

