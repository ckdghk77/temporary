/* jspsych-instructions.js
 * Josh de Leeuw
 *
 * This plugin displays text (including HTML formatted strings) during the experiment.
 * Use it to show instructions, provide performance feedback, etc...
 *
 * Page numbers can be displayed to help with navigation by setting show_page_number
 * to true.
 *
 * documentation: docs.jspsych.org
 *
 *
 */

jsPsych.plugins['instructions'] = (function() {

  var plugin = {};

  plugin.info = {
    name: 'instructions',
    description: '',
    parameters: {
      pages: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Pages',
        default: undefined,
        array: true,
        description: 'Each image of the array is the content for a single page.'
      },
      key_forward: {
        type: jsPsych.plugins.parameterType.KEY,
        pretty_name: 'Key forward',
        default: 'ArrowRight',
        description: 'The key the subject can press in order to advance to the next page.'
      },
      key_backward: {
        type: jsPsych.plugins.parameterType.KEY,
        pretty_name: 'Key backward',
        default: 'ArrowLeft',
        description: 'The key that the subject can press to return to the previous page.'
      },
      allow_backward: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Allow backward',
        default: true,
        description: 'If true, the subject can return to the previous page of the instructions.'
      },
      allow_keys: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Allow keys',
        default: true,
        description: 'If true, the subject can use keyboard keys to navigate the pages.'
      }, 
      /////// for show oneshot
      oneshot_stimulus_dimensions: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Stimulus dimensions',
        default: [256, 256],
        description: 'Stimulus dimensions in pixels [width, height]'
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
      reward_display_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Reward display duration',
        default: 2000,
        description: 'How long to display the reward (in milliseconds).'
      },
      stimuli_display_duration: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Stimuli display duration',
          default: 700,
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
      /////////
      ////////copy for linking rating
      lc_stimulus: {
        type: jsPsych.plugins.parameterType.IMAGE,
        pretty_name: 'Stimulus',
        default: [tuto_images[0],tuto_images[1],tuto_images[2]],
        description: 'The image to be displayed'
      },
      lc_stimulus_height: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Image height',
        default: null,
        description: 'Set the image height in pixels'
      },
      lc_stimulus_width: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Image width',
        default: null,
        description: 'Set the image width in pixels'
      },
      lc_min: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Min slider',
        default: 0,
        description: 'Sets the minimum value of the slider.'
      },
      lc_max: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Max slider',
        default: 100,
        description: 'Sets the maximum value of the slider',
      },
      lc_slider_start: {
				type: jsPsych.plugins.parameterType.INT,
				pretty_name: 'Slider starting value',
				default: 50,
				description: 'Sets the starting value of the slider',
			},
      lc_step: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Step',
        default: 1,
        description: 'Sets the step of the slider'
      },
      lc_labels: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name:'Labels',
        default: [],
        array: true,
        description: 'Labels of the slider.',
      },
      lc_labels_description: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name:'Labels description',
        default: [],
        array: true,
        description: 'Labels description of the slider. (start, mid, end)',
      },
      lc_color: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'label color',
        default:  'blue',
        array: false,
        description: 'color of label'
      },
      lc_slider_width: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name:'Slider width',
        default: null,
        description: 'Width of the slider in pixels.'
      },
      lc_prompt: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Prompt',
        default: null,
        description: 'Any content here will be displayed below the slider.'
      },
      lc_guide: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Prompt',
        default: null,
        description: 'Any content here will be displayed below the slider.'
      }, 
      /////////
      ////// copy for tri rating 
      tri_stimuli: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Stimuli',
        default: [tuto_images[0],tuto_images[1],tuto_images[2]],
        array: true,
        description: 'The images to be displayed'
      },
      tri_stimulus_dimensions: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Stimulus dimensions',
        default: [256, 256],
        description: 'Stimulus dimensions in pixels [width, height]'
      }, 
      tri_prompt: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Prompt',
        default: null,
        description: 'Any content here will be displayed below the slider.'
      },
      ////////
      canvas_dimensions: {
        type:jsPsych.plugins.parameterType.INT, 
        default: [window.screen.width, window.screen.height],
        description: 'The dimensions [width, height] of the html canvas on which things are drawn'
      }, 
      show_clickable_nav: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Show clickable nav',
        default: false,
        description: 'If true, then a "Previous" and "Next" button will be displayed beneath the instructions.'
      },
      show_page_number: {
          type: jsPsych.plugins.parameterType.BOOL,
          pretty_name: 'Show page number',
          default: false,
          description: 'If true, and clickable navigation is enabled, then Page x/y will be shown between the nav buttons.'
      },
      page_label: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Page label',
        default: 'Page',
        description: 'The text that appears before x/y (current/total) pages displayed with show_page_number'
      },      
      button_label_previous: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Button label previous',
        default: 'Previous',
        description: 'The text that appears on the button to go backwards.'
      },
      button_label_next: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Button label next',
        default: 'Next',
        description: 'The text that appears on the button to go forwards.'
      }
    }
  }

  plugin.trial = function(display_element, trial) {

    var current_page = 0;

    var view_history = [];

    var trial_start_time = performance.now();

    var last_page_update_time = trial_start_time;

    var html;

    var html_str = ""; // for triangle rating

    function btnListener(evt){
    	evt.target.removeEventListener('click', btnListener);
    	if(this.id === "jspsych-instructions-back"){
    		back();
    	}
    	else if(this.id === 'jspsych-instructions-next'){
    		next();
    	}
    }

    function show_current_page() {

      display_element.innerHTML = "";

      html = "<img src='"+trial.pages[current_page]+"' id = 'curr' + width='"+trial.canvas_dimensions[0]+"' height='"+trial.canvas_dimensions[1]+"'></img>"
      //var html = "<img src='"+trial.pages[current_page]+"' id = 'curr'></img>"
      //console.log(current_page)
      display_element.innerHTML = html;
      
    }

    function next() {

      if (current_page == 11) {
        jsPsych.pluginAPI.cancelAllKeyboardResponses();
        createkeyboardListener();
        add_current_tuto_to_view_history('observation');
      } else if (current_page==18) {
        jsPsych.pluginAPI.cancelAllKeyboardResponses();
        createkeyboardListener();
        add_current_tuto_to_view_history('linking rating');        
      } else if (current_page==25) {
        jsPsych.pluginAPI.cancelAllKeyboardResponses();
        createkeyboardListener();
        add_current_tuto_to_view_history('causal rating');        
      } else if (current_page==33) {
        jsPsych.pluginAPI.cancelAllKeyboardResponses();
        createkeyboardListener();
        add_current_tuto_to_view_history('bonus rating');        
      } else if (current_page==34) {
        jsPsych.pluginAPI.cancelAllKeyboardResponses();
        createkeyboardListener();
        add_current_tuto_to_view_history('one round practice');        
      } else {
        add_current_page_to_view_history();
      }

      /*
      if (current_page == 11 || current_page == 18 || current_page == 25 || current_page == 34 || current_page == 35 ){
        jsPsych.pluginAPI.cancelAllKeyboardResponses();
        createkeyboardListener();
      } 
      */

      current_page++;

      // if done, finish up...
      if (current_page >= trial.pages.length) {
        endTrial();
      } else {
        show_current_page();
      }

    }

    function back() {

      add_current_page_to_view_history()

      current_page--;

      show_current_page();
    }


    function add_current_page_to_view_history() {

      var current_time = performance.now();

      var page_view_time = current_time - last_page_update_time;

      view_history.push({
        page_index: current_page,
        viewing_time: [last_page_update_time, current_time]
      });

      last_page_update_time = current_time;
    }

    function add_current_tuto_to_view_history(step) {

      var current_time = performance.now();

      var page_view_time = current_time - last_page_update_time;

      view_history.push({
        page_index: step,
        viewing_time: [last_page_update_time, current_time]
      });

      last_page_update_time = current_time;
    }

///////// copy from oneshot.js ///////// 

    var start_time = null; // time variable
    var end_time = null;
    var msg_start_time = null;
    var msg_end_time = null;

    var all_observation = []; // for data save of observation
    var observation_timeline = []; 

    var all_practice = [];
    var practice = [];

    var createoneshot = function (){
      jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: next,
        valid_responses: [trial.key_forward],
        rt_method: 'performance',
        persist: false,
        allow_held_key: false      
      });
    }

    function showoneshot(num_trial){

      jsPsych.pluginAPI.cancelAllKeyboardResponses();

      //createoneshot();

      var new_html = '<canvas id="trial_canvas" width="'+trial.canvas_dimensions[0]+'" height="'+trial.canvas_dimensions[1]+'"></canvas>';
      display_element.innerHTML = new_html;
      var ctx = document.getElementById('trial_canvas').getContext('2d');

      drawbackground(ctx);

      var stim_interval0 = Math.floor(Math.random()*(trial.between_stimuli_duration[1]-trial.between_stimuli_duration[0])+trial.between_stimuli_duration[0]);

      var img_idx = 0
      var event_idx = 0
      image_iter(img_idx,event_idx,ctx,num_trial,stim_interval0);

    }

    function drawbackground(ctx){
      ctx.fillStyle = '#d2d2d2';
      ctx.fillRect(0, 0, trial.canvas_dimensions[0], trial.canvas_dimensions[1]);

      ctx.fillStyle = 'rgb(0,0,0)';
      ctx.font = "24px IBM Plex Sans Extrabold"
      ctx.fillText("+",trial.canvas_dimensions[0]/2-10,trial.canvas_dimensions[1]/2-12);
    }

    function drawscreen(im,ctx) {

      var img = new Image();
      img.src = im;

      img.onload = function(){
        ctx.drawImage(img, (trial.canvas_dimensions[0]/2)-(trial.oneshot_stimulus_dimensions[0]/2), (trial.canvas_dimensions[1]/2)-(trial.oneshot_stimulus_dimensions[1]/2), 
        trial.oneshot_stimulus_dimensions[0], trial.oneshot_stimulus_dimensions[1]);
      }  
      
      start_time = performance.now(); 

      jsPsych.pluginAPI.setTimeout(function() {
        drawbackground(ctx);
        end_time = performance.now();
        observation_timeline.push([start_time,end_time]);  
      }, trial.stimuli_display_duration); 
        
    }; 

    function image_iter(img_idx,event_idx,ctx,num_trial,stim_interval){
      jsPsych.pluginAPI.setTimeout(function() {
          
          img_idx += 1;
          var image_idx = tuto_images[trial.cue_allocation[img_idx-1]];

          if (img_idx <= num_trial) {
            drawscreen(image_idx,ctx);
          } else {

            if (current_page == 34){

              practice.push(observation_timeline);
              observation_timeline = [];

              trial.lc_prompt = '<p style="color:blue;">*** How much do you like this picture? ***</p>'+'<p style="color:blue;">[&larr;/&rarr;] to change, [Y] to confirm</p>';
              trial.lc_guide = null;
              trial.lc_labels_description = ["(Dislike)","(Don't know)","(Like)"];
              trial.lc_color = 'blue'
              trial.lc_min = -5
              trial.lc_max = 5
              trial.lc_slider_start = 0  
              trial.lc_labels = ['-5','-4','-3','-2','-1','0','1','2','3','4','5']
              rating_type = 'l'
              rating_num = 1
              showlinkingcausalrating(); 
              return;
            } else {

              all_observation.push(observation_timeline);
              observation_timeline = [];

              next();
              return;
            }
          }

          var stim_interval0 = Math.floor(Math.random()*(trial.between_stimuli_duration[1]-trial.between_stimuli_duration[0])+trial.between_stimuli_duration[0]);
          
          if (img_idx%5 == 0){ 
            reward_iter(img_idx,event_idx,ctx,num_trial,stim_interval0);
          }else if (img_idx<=num_trial) {
            image_iter(img_idx,event_idx,ctx,num_trial,stim_interval0); 
          } 
          
        }, (trial.stimuli_display_duration+stim_interval));
    }

    function reward_iter(img_idx,event_idx,ctx,num_trial,stim_interval){
      jsPsych.pluginAPI.setTimeout(function() {
          
          event_idx += 1;
          reward_image = reward_images[trial.reward_allocation[event_idx-1]];

          var img = new Image();
          img.src = reward_image;
          img.onload = function(){
            img_dimensions = [639, 346];
            ctx.drawImage(img, (trial.canvas_dimensions[0]/2)-(img_dimensions[0]/2), (trial.canvas_dimensions[1]/2)-(img_dimensions[1]/2), 
            img_dimensions[0], img_dimensions[1]);
          }  
          start_time = performance.now(); 

          jsPsych.pluginAPI.setTimeout(function() {
            drawbackground(ctx);
            end_time = performance.now();
            observation_timeline.push([start_time,end_time]); 
          }, trial.reward_display_duration); 
          
          var trial_interval0 = Math.floor(Math.random()*(trial.iti_duration[1]-trial.iti_duration[0])+trial.iti_duration[0]);

          jsPsych.pluginAPI.setTimeout(function(){
            image_iter(img_idx,event_idx,ctx,num_trial,trial_interval0);
          },trial.reward_display_duration-trial.stimuli_display_duration);
          
        }, (trial.stimuli_display_duration+stim_interval));
    }    

////////////////////////////////////    

///////// copy from image_slider_response.js ///////// 

    var rating_num = 1;
    var rating_type = "";
    var rating_gap_duration = null;

    var all_linking = []; //for one linking
    var linking_timeline = [];

    var all_causal = []; // for one causal
    var causal_timeline = [];

    var linking_ratings = []; // rating value , [start_time, end_time]
    var causal_ratings = [];

    function answercheck() {

      jsPsych.pluginAPI.cancelAllKeyboardResponses();

      // At page 35, show one entire round 
      if (current_page == 34 && rating_type == 'l') { // 3 linking rating then causal rating
        (document.body).removeEventListener('click',focuslc);

        var response_value = display_element.querySelector('#jspsych-image-slider-response-response').valueAsNumber;
        end_time = performance.now();
        linking_ratings.push([response_value,[start_time,end_time]]);

          if (rating_num<3){ // 3 linking rating

            rating_num++;
            html_gap = "<p style='font-familiy:IBM Plex Sans Extrabold;font-size:24px;'>+</p>";
            display_element.innerHTML = html_gap;
            rating_gap_duration = Math.floor(Math.random()*(100)+200);
            
            jsPsych.pluginAPI.setTimeout(function() {
              showlinkingcausalrating();
            }, (rating_gap_duration));

            return;
          } else { // first causal rating

            trial.lc_prompt = '<p style="color:red;">*** How much do you think this causes a significant impact on the outcome? ***</p>'+'<p style="color:red;">[&larr;/&rarr;] to change, [Y] to confirm</p>';
            trial.lc_guide = null;
            trial.lc_labels_description = ["(Not at all)","(Don't know)","(Very likely)"];
            trial.lc_color = 'red';
            trial.lc_min = 0;
            trial.lc_max = 10;
            trial.lc_slider_start = 5;
            trial.lc_labels = ['0','1','2','3','4','5','6','7','8','9','10'];
            rating_num = 1;
            rating_type = 'c';

            html_gap = "<p style='font-familiy:IBM Plex Extrabold;font-size:24px;'>+</p>";
            display_element.innerHTML = html_gap;
            rating_gap_duration = Math.floor(Math.random()*(100)+200);
            
            jsPsych.pluginAPI.setTimeout(function() {
              practice.push(linking_ratings);
              linking_ratings = [];
              showlinkingcausalrating();
            }, (rating_gap_duration));
            
            return;  
          }
      } else if (current_page == 34 && rating_type  == 'c') { // 3 causal rating then tri rating
        (document.body).removeEventListener('click',focuslc);

        var response_value = display_element.querySelector('#jspsych-image-slider-response-response').valueAsNumber;
        end_time = performance.now();
        causal_ratings.push([response_value,[start_time,end_time]]);

          if (rating_num<3){
            rating_num++;

            html_gap = "<p style='font-familiy:IBM Plex Sans Extrabold;font-size:24px;'>+</p>";
            display_element.innerHTML = html_gap;
            rating_gap_duration = Math.floor(Math.random()*(100)+200);
            
            jsPsych.pluginAPI.setTimeout(function() {
              showlinkingcausalrating();
            }, (rating_gap_duration));

            return;
          } else {
            rating_num = 1;
            rating_type = '';
            trial.tri_prompt = null;
            html_gap = "<p style='font-familiy:IBM Plex Sans Extrabold;font-size:24px;'>+</p>";
            display_element.innerHTML = html_gap;
            rating_gap_duration = Math.floor(Math.random()*(100)+200);
            
            jsPsych.pluginAPI.setTimeout(function() {
              practice.push(causal_ratings);
              causal_ratings = [];
              showtrirating();
            }, (rating_gap_duration));
            
            return;
          }        
      }

      var answer = display_element.querySelector('#jspsych-image-slider-response-response').valueAsNumber;
      if (answer == 3){
        (document.body).removeEventListener('click',focuslc);

        end_time = performance.now();
        if (rating_type == 'l'){
          linking_timeline.push([answer,[start_time,end_time]]);
          all_linking.push(linking_timeline);
          linking_timeline = [];
        } else {
          causal_timeline.push([answer,[start_time,end_time]]);
          all_causal.push(causal_timeline);
          causal_timeline = [];
        }
        
        next();
      } else {
        var msg = "<p>Try again &#x2639</p>";

        document.getElementById("everything-warraper").style.display = 'none';
        display_element.innerHTML += msg;

        msg_start_time = performance.now();

        jsPsych.pluginAPI.setTimeout(function() {
          msg_end_time = performance.now();
          if (rating_type == 'l'){
            linking_timeline.push([answer,[msg_start_time,msg_end_time]]);
          } else {
            causal_timeline.push([answer,[msg_start_time,msg_end_time]]);
          }
          display_element.innerHTML = html;
          //(document.body).getElementById("everything-warraper").style.display = 'block';
          display_element.querySelector('#jspsych-image-slider-response-response').focus()
          submitlc();
          }, 3000);        
      }

    }

    var submitlc= function() {
      jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: answercheck,
        valid_responses: ['y'],
        rt_method: 'performance',
        persist: false,
        allow_held_key: false      
      });
    }

    var focuslc = function() {
      display_element.querySelector('#jspsych-image-slider-response-response').focus()
    }


    function showlinkingcausalrating() {

      jsPsych.pluginAPI.cancelAllKeyboardResponses();

      display_element.innerHTML = "";

      submitlc();

      (document.body).addEventListener('click',focuslc);

      var height, width;
      var half_thumb_width = 7.5; 

      html = '';

      html = '<div id = "everything-warraper">' // contain everything include guide

      if (trial.lc_guide != null){
        html += trial.lc_guide;
      }

      html += '<div style="margin-top: 40px;">' // contain question and slicer wrapper
      html += trial.lc_prompt; 

      html += '<div id="jspsych-image-slider-response-wrapper" style="margin: 100px 0px;">';
      html += '<div id="jspsych-image-slider-response-stimulus">';
      html += '<img src="'+trial.lc_stimulus[rating_num-1]+'" style="';
      if(trial.lc_stimulus_height !== null){
        html += 'height:'+trial.lc_stimulus_height+'px; '
        if(trial.lc_stimulus_width == null){
          html += 'width: auto; ';
         }
      }
      if(trial.lc_stimulus_width !== null){
        html += 'width:'+trial.lc_stimulus_width+'px; '
        if(trial.lc_stimulus_height == null){
          html += 'height: auto; ';
        }
      }
      html += '"></img>';
      html += '</div>';
      html += '<div class="jspsych-image-slider-response-container" style="position:relative; margin: 0 auto 3em auto; width:';
      if (trial.lc_slider_width !== null) {
        html += trial.lc_slider_width+'px;';
      } else {
        html += 'auto;';
      }
      html += '">';
      html += '<input type="range" class="jspsych-slider" value="'+trial.lc_slider_start+'" min="'+trial.lc_min+'" max="'+trial.lc_max+'" step="'+trial.lc_step+'" id="jspsych-image-slider-response-response"></input>';
      html += '<div>'

      for(var j=0; j < trial.lc_labels.length; j++){
        var label_width_perc = 100/(trial.lc_labels.length-1);
        var percent_of_range = j * (100/(trial.lc_labels.length - 1));
        var percent_dist_from_center = ((percent_of_range-50)/50)*100;
        var offset = (percent_dist_from_center * half_thumb_width)/100;
        html += '<div style="border: 1px solid transparent; display: inline-block; position: absolute; '+
        'left:calc('+percent_of_range+'% - ('+label_width_perc+'% / 2) - '+offset+'px); text-align: center; width: '+label_width_perc+'%;">';
        if (j==0) {
          html += '<div style="color:'+trial.lc_color+';text-align: center; font-size: 80%;">'+trial.lc_labels[j]+'<br>'+trial.lc_labels_description[0]+'</div>';
        } else if (j==parseInt((trial.lc_labels.length-1)/2)){
          html += '<div style="color:'+trial.lc_color+';text-align: center; font-size: 80%;">'+trial.lc_labels[j]+'<br>'+trial.lc_labels_description[1]+'</div>';
        } else if (j==(trial.lc_labels.length-1)){
          html += '<div style="color:'+trial.lc_color+';text-align: center; font-size: 80%;">'+trial.lc_labels[j]+'<br>'+trial.lc_labels_description[2]+'</div>';
        }  else {
          html += '<div style="color:'+trial.lc_color+';text-align: center; font-size: 80%;">'+trial.lc_labels[j]+'</div>';
        }
        html += '</div>'
      }
      html += '</div>';
      html += '</div>';
      html += '</div>';

      html += '</div>'; // contain question and slicer wrapper
      html += '</div>'; // contain everything include guide
      
      display_element.innerHTML = html;

      var img = display_element.querySelector('img');
      if (trial.lc_stimulus_height !== null) {
        height = trial.lc_stimulus_height;
        if (trial.lc_stimulus_width == null) {
          width = img.naturalWidth * (trial.lc_stimulus_height/img.naturalHeight);
        }
      } else {
        height = img.naturalHeight;
      }
      if (trial.lc_stimulus_width !== null) {
        width = trial.lc_stimulus_width;
        if (trial.lc_stimulus_height == null) {
          height = img.naturalHeight * (trial.lc_stimulus_width/img.naturalWidth);
        }
      } else if (!(trial.lc_stimulus_height !== null)) {
        // if stimulus width is null, only use the image's natural width if the width value wasn't set 
        // in the if statement above, based on a specified height and maintain_aspect_ratio = true
        width = img.naturalWidth;
      }
      img.style.height = height.toString() + "px";
      img.style.width = width.toString() + "px";

      // focus without click slider
      display_element.querySelector('#jspsych-image-slider-response-response').focus();

      start_time = performance.now();
      //linking_timeline.push([start_time,null]);
      
    }

//////////////////////////////////// 

///////// copy from oneshot_triangle.js ///////// 

    all_tri = []; // for one tri rating
    tri_timeline = [];

    tri_ratings = []; //value 0, value 1, value 2, [start_time, end_time]

    var answercheck_tri = function(){

      jsPsych.pluginAPI.cancelAllKeyboardResponses();

      if (current_page == 34){
        tri_ratings = barycentric_system(c_idx);
        end_time = performance.now();
        tri_ratings.push([start_time,end_time]);
        practice.push(tri_ratings);
        tri_ratings=[];
        all_practice.push(practice);
        practice = [];
        next();
        return;
      }

      var true_answer = [0.25,0.75,0];
      var answer = barycentric_system(c_idx);

      //if (answer[0] == true_answer[0] && answer[1] == true_answer[1] && answer[2] == true_answer[2]){
      if (c_idx == 11)
      {
        end_time = performance.now();
        tri_timeline.push([answer,[start_time,end_time]]);
        all_tri.push(tri_timeline);
        tri_timeline=[];

        next();
      } else {
        var msg = "<p>Try again &#x2639</p>";

        display_element.innerHTML = msg;

        msg_start_time = performance.now();

        jsPsych.pluginAPI.setTimeout(function() {
          msg_end_time = performance.now();
          tri_timeline.push([answer,[msg_start_time,msg_end_time]]);
          display_element.innerHTML = html_str;
          display_element.querySelector('#C'+c_idx).className = 'plus_selected'
          submittri();
          }, 3000);        
      }

    }

    var tri_response = function(info) {

      next_idxs = next_circle[c_idx]
      submittri();

        if (info.key === 'arrowleft') {
          new_idx = next_idxs[1]
          if (new_idx !=0) {
            highlight_circle(new_idx,c_idx)
            showvalues(new_idx)
            c_idx = new_idx 
          }
        } else if (info.key === 'arrowright') {
         new_idx = next_idxs[3]
          if (new_idx !=0) {
            highlight_circle(new_idx,c_idx)
            showvalues(new_idx)
            c_idx = new_idx 
          }     
        } else if (info.key === 'arrowup') {
          new_idx = next_idxs[0]
          if (new_idx !=0) {
            highlight_circle(new_idx,c_idx)
            showvalues(new_idx)
            c_idx = new_idx 
          }
        } else if (info.key === 'arrowdown') {
          new_idx = next_idxs[2]
          if (new_idx !=0) {
            highlight_circle(new_idx,c_idx)
            showvalues(new_idx)
            c_idx = new_idx 
          }       

        } else if (info.key === 'y'){
          answercheck_tri();
        }
        
      //console.log(c_idx)
      };


    var submittri= function() {
      jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: tri_response,
        valid_responses: ['arrowleft', 'arrowright', 'arrowup','arrowdown', 'y'],
        rt_method: 'performance',
        persist: false,
        allow_held_key: false  
      });
    }

    function showtrirating(){

      jsPsych.pluginAPI.cancelAllKeyboardResponses();
      display_element.innerHTML = "";
      submittri();

      html_str = '';

      if (trial.tri_prompt != null){
        html_str += trial.tri_prompt;
      }

      html_str += '<div>'; // wrapper for everything below guide
      html_str += '<p style="color:green;">*** Submit relative causal rating ***</p>'+'<p style="color:green;">[&larr;/&rarr;/&uarr;/&darr;] to change, [Y] to confirm</p>'

      cur_idx = 0
      size_triangle = 110
      draw_space = 30
    //  xpos = window.screen.width/2
    //  ypos = window.screen.height/2
      
      all_xpos = [...Array(25).keys()]
      all_ypos = [...Array(25).keys()]

      total_size = 3*size_triangle+2*trial.tri_stimulus_dimensions[1];
      html_str += '<div id="triangle_wrapper" style="position:relative;width:'+(window.screen.width)+'px;height:'+total_size +'px">';

      xpos = window.screen.width/2;
      ypos = (total_size/2+draw_space);

      // create triangle lines
      line_length = size_triangle*Math.sin(Math.PI/3)*2+10
      create_line();

      // creat three points and center
      create_circle(1,xpos,ypos-size_triangle)
      create_circle(12,xpos,ypos)
      create_circle(21,xpos-size_triangle*Math.sin(Math.PI/3),ypos+size_triangle*Math.cos(Math.PI/3))
      create_circle(25,xpos+size_triangle*Math.sin(Math.PI/3),ypos+size_triangle*Math.cos(Math.PI/3))
      
      // creat vertical line
      create_circle(4,xpos,ypos-size_triangle/2)
      create_circle(7,xpos,ypos-size_triangle/4)
      create_circle(17,xpos,ypos+size_triangle/4)
      create_circle(2,xpos,ypos-3*size_triangle/4)
      create_circle(23,xpos,ypos+size_triangle/2)
      
      th_ang = Math.PI/3-Math.atan(Math.cos(Math.PI/6))
      create_circle(3,xpos-(Math.sqrt(7)/4)*size_triangle*Math.sin(th_ang),ypos-(Math.sqrt(7)/4)*size_triangle*Math.cos(th_ang))
      create_circle(5,xpos+(Math.sqrt(7)/4)*size_triangle*Math.sin(th_ang),ypos-(Math.sqrt(7)/4)*size_triangle*Math.cos(th_ang))
      create_circle(6,xpos-size_triangle*Math.tan(Math.PI/3)/4,ypos-size_triangle/4)
      create_circle(8,xpos+size_triangle*Math.tan(Math.PI/3)/4,ypos-size_triangle/4)      
      create_circle(9,xpos-size_triangle*Math.cos(Math.PI/6)/4,ypos-size_triangle*Math.sin(Math.PI/6)/4)   
      create_circle(10,xpos+size_triangle*Math.cos(Math.PI/6)/4,ypos-size_triangle*Math.sin(Math.PI/6)/4)         
      
      th_ang = Math.atan(Math.cos(Math.PI/6))-Math.PI/6
      create_circle(11,xpos-(Math.sqrt(7)/4)*size_triangle*Math.cos(th_ang),ypos+(Math.sqrt(7)/4)*size_triangle*Math.sin(th_ang))
      create_circle(13,xpos+(Math.sqrt(7)/4)*size_triangle*Math.cos(th_ang),ypos+(Math.sqrt(7)/4)*size_triangle*Math.sin(th_ang))
      create_circle(14,xpos-size_triangle*Math.cos(Math.PI/6)/4,ypos+size_triangle*Math.sin(Math.PI/6)/4)  
      create_circle(15,xpos+size_triangle*Math.cos(Math.PI/6)/4,ypos+size_triangle*Math.sin(Math.PI/6)/4)    
      create_circle(16,xpos-size_triangle*Math.cos(Math.PI/6)/2,ypos+size_triangle/4)
      create_circle(18,xpos+size_triangle*Math.cos(Math.PI/6)/2,ypos+size_triangle/4) 
      create_circle(19,xpos-size_triangle*3*Math.cos(Math.PI/6)/4,ypos+size_triangle*3/8)
      create_circle(20,xpos+size_triangle*3*Math.cos(Math.PI/6)/4,ypos+size_triangle*3/8)   
      create_circle(22,xpos-size_triangle*Math.sin(Math.PI/3)+size_triangle*Math.cos(Math.PI/6)/2,ypos+size_triangle*Math.cos(Math.PI/3))
      create_circle(24,xpos+size_triangle*Math.sin(Math.PI/3)-size_triangle*Math.cos(Math.PI/6)/2,ypos+size_triangle*Math.cos(Math.PI/3))

      next_circle = [...Array(26).keys()]
      next_circle[0] = [0,0,0,0]

      next_circle[1] = [0,0,2,0]
      next_circle[2] = [1,3,4,5]
      next_circle[3] = [1,0,9,2]
      next_circle[5] = [1,2,10,0]
      next_circle[4] = [2,3,7,5]

      next_circle[6] = [3,0,16,9]
      next_circle[8] = [5,10,18,0]
      next_circle[7] = [4,9,12,10]
      next_circle[9] = [3,6,14,7]
      next_circle[10] = [5,7,15,8]

      next_circle[12] = [7,14,17,15] 
      next_circle[11] = [6,0,19,16] 
      next_circle[13] = [8,18,20,0]
      next_circle[14] = [9,16,0,17]
      next_circle[15] = [10,17,0,18]
      
      next_circle[16] = [6,11,22,14]
      next_circle[17] = [12,14,23,15]
      next_circle[18] = [8,15,24,13]   
      next_circle[19] = [11,21,0,22] 
      next_circle[20] = [13,24,0,25] 

      next_circle[21] = [11,0,0,22] 
      next_circle[22] = [16,21,0,23]
      next_circle[23] = [17,22,0,24]
      next_circle[24] = [18,23,0,25]
      next_circle[25] = [13,24,0,0]

      // display stimulus images
      drawstimuli();

      // display arrow mark
      if (trial.tri_prompt != null){
        // idx : 11
        html_str += "<div id='mark' style='color:red;position:absolute;top:"+ (ypos+(Math.sqrt(7)/4)*size_triangle*Math.sin(th_ang)-10)+
         "px; left: " + (xpos-(Math.sqrt(7)/4)*size_triangle*Math.cos(th_ang)-20) +"px'><b>&#10140</b></div>";
      }

      // initial setting
      init_idx = 12;
      c_idx = init_idx;
      init_values = barycentric_system(c_idx)

      html_str +=  "<p id='A' style='color:green;position:absolute;top:"+ (ypos-size_triangle-trial.tri_stimulus_dimensions[1]-draw_space*3)+ "px; left: " 
      + (xpos-(trial.tri_stimulus_dimensions[0]/2)+draw_space*2) +"px'>"+init_values[0].toFixed(2)+"</p>";
      html_str +=  "<p id='B' style='color:green;position:absolute;top:"+ (ypos+size_triangle*Math.cos(Math.PI/3)+trial.tri_stimulus_dimensions[1]+draw_space)+ "px; left: " 
      + (xpos-size_triangle*Math.sin(Math.PI/3)-trial.tri_stimulus_dimensions[0]+draw_space*1.5) +"px'>"+init_values[1].toFixed(2)+"</p>";
      html_str +=  "<p id='C' style='color:green;position:absolute;top:"+ (ypos+size_triangle*Math.cos(Math.PI/3)+trial.tri_stimulus_dimensions[1]+draw_space)+ "px; left: " 
      + (xpos+size_triangle*Math.sin(Math.PI/3)+draw_space*2.7) +"px'>"+init_values[2].toFixed(2)+"</p>";

      html_str += '</div>';
      //html_str += '<p style="color:blue;">[y]: submit, [n]: reset</p>';
      
      html_str += '</div>'; // wrapper for everything below guide
      
      display_element.innerHTML = html_str;

      display_element.querySelector('#C'+init_idx).className = 'plus_selected'

      start_time = performance.now();

    }

    function drawstimuli(){
      html_str += '<img src="'+trial.tri_stimuli[0]+'" id="image_0"'+'width="'+trial.tri_stimulus_dimensions[0]+'"height="'+trial.tri_stimulus_dimensions[1]+
      '"style="position:absolute;top:'+(ypos-size_triangle-trial.tri_stimulus_dimensions[1]-draw_space)+'px; left: '+(xpos-(trial.tri_stimulus_dimensions[0]/2)+10)+'px;"></img>'
      html_str += '<img src="'+trial.tri_stimuli[1]+'" id="image_1"'+'width="'+trial.tri_stimulus_dimensions[0]+'"height="'+trial.tri_stimulus_dimensions[1]+
      '"style="position:absolute;top:'+(ypos+size_triangle*Math.cos(Math.PI/3)+draw_space)+'px; left: '+(xpos-size_triangle*Math.sin(Math.PI/3)-trial.tri_stimulus_dimensions[0]-(draw_space-20))+'px;"></img>'
      html_str += '<img src="'+trial.tri_stimuli[2]+'" id="image_2"'+'width="'+trial.tri_stimulus_dimensions[0]+'"height="'+trial.tri_stimulus_dimensions[1]+ 
      '"style="position:absolute;top:'+(ypos+size_triangle*Math.cos(Math.PI/3)+draw_space)+'px; left: '+(xpos+size_triangle*Math.sin(Math.PI/3)+draw_space)+'px;"></img>'
    }

    function create_line(){
      html_str +=  "<div class='line' id='line_0' style='height:"+line_length+"px ;transform:rotate(-30deg);top:"
      +(ypos-size_triangle/4+4-line_length/2)+"px; left:"+(xpos+4+size_triangle*Math.tan(Math.PI/3)/4)+"px'></div>";
      html_str +=  "<div class='line' id='line_1' style='height:"+line_length+"px ;transform:rotate(30deg);top:"
      +(ypos-size_triangle/4+4-line_length/2)+"px; left:"+(xpos+4-size_triangle*Math.tan(Math.PI/3)/4)+"px'></div>";
      html_str +=  "<div class='line' id='line_2' style='height:"+line_length+"px ;transform:rotate(90deg);top:"
      +(ypos+size_triangle*Math.cos(Math.PI/3)+4-line_length/2)+"px; left:"+(xpos+4)+"px'></div>";
      html_str +=  "<div class='line' id='line_3' style='height:"+(size_triangle+size_triangle*Math.cos(Math.PI/3))+"px ;top:"
      +(ypos-size_triangle+4)+"px; left:"+(xpos+4)+"px'></div>";
      html_str +=  "<div class='line' id='line_4' style='height:"+(size_triangle+size_triangle*Math.cos(Math.PI/3))+"px ;transform:rotate(-60deg);transform-origin : 0% 0%; top:"
      +(ypos-size_triangle/4+4)+"px; left:"+(xpos+4-size_triangle*Math.tan(Math.PI/3)/4)+"px'></div>";
      html_str +=  "<div class='line' id='line_5' style='height:"+(size_triangle+size_triangle*Math.cos(Math.PI/3))+"px ;transform:rotate(60deg);transform-origin : 0% 0%; top:"
      +(ypos-size_triangle/4+4)+"px; left:"+(xpos+size_triangle*Math.tan(Math.PI/3)/4+4)+"px'></div>";          
  }

    function create_circle(c_idx,left_pos,top_pos){
        all_xpos[c_idx-1] = left_pos;
        all_ypos[c_idx-1] = top_pos;
        html_str +=  "<div class='plus' id='C" + c_idx + "' style='position:absolute;top:"+ top_pos+ "px; left: " + left_pos +"px'></div>";
      
    }

    function highlight_circle(new_idx, last_idx){
        display_element.querySelector('#C'+new_idx).className = 'plus_selected'
        display_element.querySelector('#C'+last_idx).className = 'plus'
    }

    // three point idxes: 1, 21, 25
    // calculate barycentric coordinate
    function barycentric_system(c_idx){
      xpos_p = all_xpos[c_idx-1]
      ypos_p = all_ypos[c_idx-1]
      denorm = (all_ypos[20]-all_ypos[24])*(all_xpos[0]-all_xpos[24])+(all_xpos[24]-all_xpos[20])*(all_ypos[0]-all_ypos[24])
      b_value = ((all_ypos[20]-all_ypos[24])*(xpos_p-all_xpos[24])+(all_xpos[24]-all_xpos[20])*(ypos_p-all_ypos[24]))/denorm
      b_value = Math.max(b_value,0)
      b_value = Math.min(b_value,1)
      c_value = ((all_ypos[24]-all_ypos[0])*(xpos_p-all_xpos[24])+(all_xpos[0]-all_xpos[24])*(ypos_p-all_ypos[24]))/denorm
      c_value = Math.max(c_value,0)
      c_value = Math.min(c_value,1)
      a_value = 1-b_value-c_value
      a_value = Math.abs(a_value)
      b_value = Math.abs(b_value)
      c_value = Math.abs(c_value) 

      return [b_value,c_value,a_value]
    }

    function showvalues(c_idx){
      values = barycentric_system(c_idx)
      display_element.querySelector('#A').innerHTML = values[0].toFixed(2)
      display_element.querySelector('#B').innerHTML = values[1].toFixed(2)
      display_element.querySelector('#C').innerHTML = values[2].toFixed(2)
    }


//////////////////////////////////// 

    function endTrial() {

      if (trial.allow_keys) {
        jsPsych.pluginAPI.cancelKeyboardResponse(keyboard_listener);
      }

      jsPsych.pluginAPI.cancelAllKeyboardResponses();

      display_element.innerHTML = '';

      var trial_data = {
        view_history: view_history,
        rt: performance.now() - trial_start_time,
        observation: all_observation,
        linking_rating: all_linking,
        causal_rating: all_causal,
        tri_rating: all_tri,
        practice: all_practice,
      };

      jsPsych.finishTrial(trial_data);
    }

    var after_response = function(info) {
      // have to reinitialize this instead of letting it persist to prevent accidental skips of pages by holding down keys too long
      keyboard_listener = jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: after_response,
        valid_responses: [trial.key_forward, trial.key_backward],
        rt_method: 'performance',
        persist: false,
        allow_held_key: false
      });
      // check if key is forwards or backwards and update page
      if (jsPsych.pluginAPI.compareKeys(info.key, trial.key_backward)) {
        if (current_page !== 0 && trial.allow_backward && current_page != 35) {
          back();
        }
      }
      if (jsPsych.pluginAPI.compareKeys(info.key, trial.key_forward)) {
        if (current_page == 11){
          add_current_page_to_view_history();
          showoneshot(5);
        } else if (current_page == 18){
          add_current_page_to_view_history();
          trial.lc_prompt = '<p style="color:blue;">*** How much do you like this picture? ***</p>'+'<p style="color:blue;">[&larr;/&rarr;] to change, [Y] to confirm</p>'
          trial.lc_guide = "<p style='font-size:25px;color:blue;'><b>Try moving the cursor to ‘3’ using left/right arrow key then submit your ratings!</b></p>";
          trial.lc_labels_description = ["(Dislike)","(Don't know)","(Like)"];
          trial.lc_color = 'blue';
          rating_type = 'l';
          showlinkingcausalrating();
        } else if (current_page == 25){
          add_current_page_to_view_history();
          trial.lc_prompt = '<p style="color:red;">*** How much do you think this causes a significant impact on the outcome? ***</p>'+'<p style="color:red;">[&larr;/&rarr;] to change, [Y] to confirm</p>';
          trial.lc_guide = "<p style='font-size:25px;color:red;'><b>Try moving the cursor to ‘3’ using left/right arrow key then submit your ratings!</b></p>";
          trial.lc_labels_description = ["(Not at all)","(Don't know)","(Very likely)"];
          trial.lc_color = 'red';
          trial.lc_min = 0;
          trial.lc_max = 10;
          trial.lc_slider_start = 5;
          trial.lc_labels = ['0','1','2','3','4','5','6','7','8','9','10'];
          rating_type = 'c';
          showlinkingcausalrating();
        } else if (current_page == 33){
          add_current_page_to_view_history();
          trial.tri_prompt = '<p style="font-size:25px;color:green;"><b>Try moving the cursor to a pointed position using the left/right/up/down arrow key then submit your ratings!</b></p>'
          showtrirating();
        } else if (current_page == 34){
          add_current_page_to_view_history();
          showoneshot(25); // one round         
        }
        else {
          next();
        }
      }
    }; 

    show_current_page();
    createkeyboardListener();

    function createkeyboardListener() {
      var keyboard_listener = jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: after_response,
        valid_responses: [trial.key_forward, trial.key_backward],
        rt_method: 'performance',
        persist: false
      });
    }

    /*   
    if (trial.allow_keys) {
      var keyboard_listener = jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: after_response,
        valid_responses: [trial.key_forward, trial.key_backward],
        rt_method: 'performance',
        persist: false
      });
    } */
  };

  return plugin;
})();
