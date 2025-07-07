jsPsych.plugins['os_random'] = (function() {

    var plugin = {};
  
    // ask jsPsych to preload the images
    jsPsych.pluginAPI.registerPreload('moat', 'cue_images', 'image');
    jsPsych.pluginAPI.registerPreload('moat', 'reward_images', 'image');
  
    plugin.info = {
      name: 'os_random',
      parameters: {
        os_cues: {
          type:jsPsych.plugins.parameterType.STRING, // BOOL, STRING, INT, FLOAT, FUNCTION, KEYCODE, SELECT, HTML_STRING, IMAGE, AUDIO, VIDEO, OBJECT, COMPLEX
          default: null,
          description: 'The array of paths to cue images'
        },
        os_rewards: {
          type:jsPsych.plugins.parameterType.STRING, // BOOL, STRING, INT, FLOAT, FUNCTION, KEYCODE, SELECT, HTML_STRING, IMAGE, AUDIO, VIDEO, OBJECT, COMPLEX
          default: null,
          description: 'The array of paths to reward images'
        },
        cue_allocation: {
          type:jsPsych.plugins.parameterType.INT, // BOOL, STRING, INT, FLOAT, FUNCTION, KEYCODE, SELECT, HTML_STRING, IMAGE, AUDIO, VIDEO, OBJECT, COMPLEX
          default: [0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
          description: 'The mapping to subject-specific images (allows randomisation of visual stimuli)'
        },
        reward_allocation: {
            type:jsPsych.plugins.parameterType.INT, // BOOL, STRING, INT, FLOAT, FUNCTION, KEYCODE, SELECT, HTML_STRING, IMAGE, AUDIO, VIDEO, OBJECT, COMPLEX
            default: [0, 0, 0, 0, 1],
            description: 'The mapping to subject-specific images (allows randomisation of visual stimuli)'
        },
        canvas_dimensions: {
          type:jsPsych.plugins.parameterType.INT, // BOOL, STRING, INT, FLOAT, FUNCTION, KEYCODE, SELECT, HTML_STRING, IMAGE, AUDIO, VIDEO, OBJECT, COMPLEX
          default: [window.screen.width, window.screen.height],//[1000, 500],
          description: 'The dimensions [width, height] of the html canvas on which things are drawn'
        },
        background_colour: {
          type: jsPsych.plugins.parameterType.STRING,
          default: '#d2d2d2',//'#878787',
          description: 'The colour of the background'
        },
        stimulus_offset: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Stimulus offset',
          default: [0, 0],
          description: 'The offset [horizontal, vertica] of the centre of each stimulus from the centre of the canvas in pixels'
        },
        stimulus_dimensions: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Stimulus dimensions',
          default: [256, 256],
          description: 'Stimulus dimensions in pixels [width, height]'
        },
        left_key: {
          type: jsPsych.plugins.parameterType.KEYCODE,
          pretty_name: 'Left key',
          default: 'arrowleft',
          description: 'The key to be pressed to select the left planet'
        },
        right_key: {
          type: jsPsych.plugins.parameterType.KEYCODE,
          pretty_name: 'Right key',
          default: 'arrowright',
          description: 'The key to be pressed to select the right planet'
        },
        choice_listen_duration: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Choice window duration',
          default: 100000,
          description: 'How long to wait for a response (in milliseconds).'
        },
        choice_display_duration: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Choice display duration',
          default: 1500,
          description: 'How long to display the response (in milliseconds).'
        },
        reward_display_duration: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Reward display duration',
          default: 500,
          description: 'How long to display the reward (in milliseconds).'
        },
        stimuli_display_duration: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Stimuli display duration',
            default: 500,
            description: 'How long to display the reward (in milliseconds).'
          },
        iti_duration: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Duration of inter-trial interval',
          default: [1000, 2000],
          description: 'How long to display a blank screen between trials (in milliseconds).'
        },
        between_stimuli_duration: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Duration between stimulus and next stimulus',
          default: [200, 300],
          description: 'How long to display a blank screen between stimuli (in milliseconds).'
        },        
        selection_pen_width: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Width of selection box',
          default: 15,
          description: 'Thickness (in pixels) of the selection box'
        },
        selection_colour: {
          type: jsPsych.plugins.parameterType.STRING,
          default: '#FFFFFF', // #FFFFFF is white
          description: 'The colour of the selection box'
        },
        reward_offset: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Reward offset',
          default: [0, 0],//[270, -200],
          description: 'The offset [horizontal, vertica] of the centre of the reward from the centre of the canvas in pixels'
        },
        reward_dimensions: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Reward dimensions',
          default: [639, 346],
          description: 'Reward image dimensions in pixels [width, height]'
        },
        
      }
    }
  
    plugin.trial = function(display_element, trial) {
  
      // add a canvas to the HTML_STRING, store its context, and draw a blank background
      var new_html = '<canvas id="trial_canvas" width="'+trial.canvas_dimensions[0]+'" height="'+trial.canvas_dimensions[1]+'"></canvas>';
      display_element.innerHTML = new_html;
      var ctx = document.getElementById('trial_canvas').getContext('2d');
      DrawBackground(); // draw the background of the canvas
  
      // set up a container for key responses
      var response = {
        stim_and_reward: [],
        stim_timeline: [],
        description: null,
      };
  
      // set up a container for display configuration
      var display = {
        image : null
      }

      // for slider response
      
      ///// TRIAL LOOP /////
      
      var img_idx = 0
      var event_idx = 0
      
      // get random allocation
      //trial.cue_allocation = alloc_infos.cue_allocation
      //trial.reward_allocation = alloc_infos.reward_allocation

      var one_row = [];
      var one_row_rt = [];
      var start_time = null;
      var end_time = null;  

      // select random stim interval in between
      var stim_interval0 = Math.floor(Math.random()*(trial.between_stimuli_duration[1]-trial.between_stimuli_duration[0])+trial.between_stimuli_duration[0]);
      image_iter(stim_interval0);

      /*
      for (row=0; row < n_rows; row++)
      {
        for (col=0; col < n_cols; col++)
        {
            display.image_idx = cue_images[trial.cue_allocation[row*n_rows + col]];
            
            console.log("Draw Screen " + row +" " + col)
            DrawScreen();
            // set a timeout to end the trial after a given delay
            
            jsPsych.pluginAPI.setTimeout(function() {
                ITI();
            }, trial.stimuli_display_duration);
        }

        display.image_idx = reward_images[trial.reward_allocation[row]];
        jsPsych.pluginAPI.setTimeout(function() {
            ITI();
        }, trial.reward_display_duration);
        
      }*/
     
      ///// MAIN FUNCTIONS /////
  
      // function to draw background
      function DrawBackground(){

        // draw the background
        ctx.fillStyle = trial.background_colour;
        ctx.fillRect(0, 0, trial.canvas_dimensions[0], trial.canvas_dimensions[1]);

        ctx.fillStyle = 'rgb(0,0,0)';
        ctx.font = "24px Open Sans Extrabold"
        ctx.fillText("+",trial.canvas_dimensions[0]/2-10,trial.canvas_dimensions[1]/2-12)
        //ctx.fillRect(trial.canvas_dimensions[0]/2-5,trial.canvas_dimensions[1]/2+1,10,2);
        //ctx.fillRect(trial.canvas_dimensions[0]/2-1,trial.canvas_dimensions[1]/2-3,2,10);

        // draw the progress text
        //ctx.font = "28px Arial";
        //ctx.fillStyle = "white";
        //ctx.textAlign = "center";
        //var info_text = "  ";
        //ctx.fillText(info_text, trial.canvas_dimensions[0]/2, 3* ctx.measureText('M').width/2);
  
  
      }; // end DrawBackground function
  
      // function to draw the stimuli to screen
      function DrawScreen() {

        // draw the samples (novel or non-novel) after a delay
        _DrawStimulus(display.image_idx, [-trial.stimulus_offset[0], trial.stimulus_offset[1]]);
        start_time = performance.now();          

        // hide stimulus to distinguish each stimulus 
        jsPsych.pluginAPI.setTimeout(function() {
          DrawBackground();
          end_time = performance.now();
          one_row_rt.push([start_time,end_time]);        

        }, trial.stimuli_display_duration); 
          
      }; // end DrawScreen function
     
      // function to draw next Image
      function image_iter(stim_interval){
        jsPsych.pluginAPI.setTimeout(function() {
            img_idx += 1;
            //console.log(img_idx)

            one_row.push(trial.cue_allocation[img_idx-1]+1);
            display.image_idx = cue_images[trial.cue_allocation[img_idx-1]];
            
            // After sequence of 5 images, show outcome then draw next image
            if (img_idx > 25) {
              jsPsych.pluginAPI.setTimeout(function() {
                EndTrial();
              }, trial.iti_duration[1]);
            } else {
              DrawScreen();
            }

            var stim_interval0 = Math.floor(Math.random()*(trial.between_stimuli_duration[1]-trial.between_stimuli_duration[0])+trial.between_stimuli_duration[0]);
            
            if (img_idx%5 == 0){ 
              reward_iter(stim_interval0);
            }else if (img_idx<=25) {
              image_iter(stim_interval0); 
            } 
            
          }, (trial.stimuli_display_duration+stim_interval));
      }

      // function to show reward image after 5 sample images
      function reward_iter(stim_interval){
        jsPsych.pluginAPI.setTimeout(function() {
            event_idx += 1;
            //console.log(event_idx)
            reward_image = reward_images[trial.reward_allocation[event_idx-1]];
            
            one_row.push(reward_images_name[trial.reward_allocation[event_idx-1]]);
            
            (response.stim_and_reward).push(one_row);
            one_row = [];

            _DrawFeedback(reward_image);
            start_time = performance.now();

            jsPsych.pluginAPI.setTimeout(function() {
              DrawBackground();
              end_time = performance.now();
              one_row_rt.push([start_time,end_time]);
              (response.stim_timeline).push(one_row_rt);
              one_row_rt = [];
            }, trial.reward_display_duration); 

            var trial_interval0 = Math.floor(Math.random()*(trial.iti_duration[1]-trial.iti_duration[0])+trial.iti_duration[0]);
            
            jsPsych.pluginAPI.setTimeout(function(){
              image_iter(trial_interval0);
            },trial.reward_display_duration-trial.stimuli_display_duration);
            
          }, (trial.stimuli_display_duration+stim_interval));

      }      
       
      // function to show an empty screen for the duration of the inter-trial interval
      function ITI() {
  
        // draw the background of the canvas
        DrawBackground();
  
        // clear keyboard listener
        jsPsych.pluginAPI.cancelAllKeyboardResponses();
  
        // kill any remaining setTimeout handlers
        jsPsych.pluginAPI.clearAllTimeouts();
  
        // set a timeout to end the ITI after a given delay
        jsPsych.pluginAPI.setTimeout(function() {
          EndTrial();
        }, trial.iti_duration);
  
      }; // end EndTrial function
  
      // function to end trial when it is time
      function EndTrial() {
  
        // clear keyboard listener
        jsPsych.pluginAPI.cancelAllKeyboardResponses();
  
        // kill any remaining setTimeout handlers
        jsPsych.pluginAPI.clearAllTimeouts();

        // gather the data to store for the trial
        var trial_data = {
          session: parseInt(repetition_count/8),
          round:(repetition_count%8),
          stim_and_reward: response.stim_and_reward,
          stim_timeline: response.stim_timeline,
          description: []
        };
        /*
          'trial_type': trial.choice_type,
          'left_image_number': display.left_image_number,
          'right_image_number': display.right_image_number,
          'left_image_type': trial.left_image_type,
          'right_image_type': trial.right_image_type,
          'ur_left_image': trial.image_allocation.findIndex(function(element, index, arr){return element == this}, display.left_image_number),
          'ur_right_image': trial.image_allocation.findIndex(function(element, index, arr){return element == this}, display.right_image_number),
          'chosen_image': response.chosen_image,
          'ur_chosen_image': response.ur_chosen_image,
          'rt': response.rt,
          'key_char': response.key_char,
          'choice': response.choice,
          'stimulus_array': [trial.left_image_number, trial.right_image_number],
          'feedback': response.feedback */
        
        // increment the trial counter
        counter.trial += 1;
  
        // move on to the next trial
        jsPsych.finishTrial(trial_data);
  
      }; // end EndTrial function
  
      function _DrawSelectionBox(stimulus_horiz_offset, stimulus_vert_offset, colour) {
  
        var selection_horiz_loc = (trial.canvas_dimensions[0]/2) + stimulus_horiz_offset  - (trial.stimulus_dimensions[0] / 2) - trial.selection_pen_width;
        var stim_vert_loc = (trial.canvas_dimensions[1]/2) + stimulus_vert_offset  - (trial.stimulus_dimensions[1] / 2) - trial.selection_pen_width; // specifies the y coordinate of the top left corner of the stimulus
  
        ctx.fillStyle = colour;
        ctx.fillRect(selection_horiz_loc, stim_vert_loc, trial.stimulus_dimensions[0] + (2 * trial.selection_pen_width), trial.stimulus_dimensions[1] + (2 * trial.selection_pen_width));
  
      } // end _DrawSelectionBox function

      function _DrawStimulus(stimulus_array, stimulus_offset) {
  
        // create new image element
        var img = new Image();
  
        // specify that the image should be drawn once it is loaded
        img.onload = function(){_ImageOnload(img, trial.stimulus_dimensions, stimulus_offset)};

        // set the source path of the image; in JavaScript, this command also triggers the loading of the image
        img.src = stimulus_array;
  
      } // end _DrawStimulus function
  
      function _DrawBlankStimulus(stimulus_offset) {
  
        var stim_horiz_loc = (trial.canvas_dimensions[0]/2) + stimulus_offset[0]  - (trial.stimulus_dimensions[0] / 2); // specifies the x coordinate of the top left corner of the stimulus
        var stim_vert_loc = (trial.canvas_dimensions[1]/2) + stimulus_offset[1] - (trial.stimulus_dimensions[1] / 2); // specifies the y coordinate of the top left corner of the stimulus
  
        ctx.fillStyle = trial.background_colour;
        ctx.fillRect(stim_horiz_loc, stim_vert_loc, trial.stimulus_dimensions[0], trial.stimulus_dimensions[1]);
  
      } // end _DrawBlankStimulus function
  
      function _DrawFeedback(feedback_image) {
  
        // create new image element
        var img = new Image();
  
        // specify that the image should be drawn once it is loaded
        img.onload = function(){
          _ImageOnload(img, trial.reward_dimensions, trial.reward_offset)
        };
  
        // set the source path of the image; in JavaScript, this command also triggers the loading of the image
        img.src = feedback_image;
  
      } // end _DrawFeedback function
  
      function _ImageOnload(im, image_dimensions, image_offset){
  
        var stim_horiz_loc = (trial.canvas_dimensions[0]/2) + image_offset[0]  - (image_dimensions[0] / 2); // specifies the x coordinate of the top left corner of the stimulus
        var stim_vert_loc = (trial.canvas_dimensions[1]/2) + image_offset[1] - (image_dimensions[1] / 2); // specifies the y coordinate of the top left corner of the stimulus
        ctx.drawImage(im, stim_horiz_loc, stim_vert_loc, image_dimensions[0], image_dimensions[1]);
  
      } // end _StimulusOnload function
  
    } // end plugin.trial
  
    return plugin;
  
  })(); // end plugin function


