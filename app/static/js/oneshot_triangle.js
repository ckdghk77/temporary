jsPsych.plugins['os_tri_question'] = (function() {

    var plugin = {};
  
    // ask jsPsych to preload the images
    jsPsych.pluginAPI.registerPreload('moat', 'notes', 'image');
  
    plugin.info = {
      name: 'os_tri_question',
      parameters: {
        notes: {
          type: jsPsych.plugins.parameterType.IMAGE,
          pretty_name: 'Notes',
          default: undefined,
          description: 'The image to be displayed'
        },
        stimuli: {
          type: jsPsych.plugins.parameterType.STRING,
          pretty_name: 'Stimuli',
          default: undefined,
          array: true,
          description: 'The images to be displayed'
        },
        stimulus_dimensions: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Stimulus dimensions',
          default: [256, 256],
          description: 'Stimulus dimensions in pixels [width, height]'
        },        
        canvas_dimensions: {
          type:jsPsych.plugins.parameterType.INT, // BOOL, STRING, INT, FLOAT, FUNCTION, KEYCODE, SELECT, HTML_STRING, IMAGE, AUDIO, VIDEO, OBJECT, COMPLEX
          default: [window.screen.width, window.screen.height],//[1000, 500],
          description: 'The dimensions [width, height] of the html canvas on which things are drawn'
        },
        background_colour: {
          type: jsPsych.plugins.parameterType.STRING,
          default: '#878787',
          description: 'The colour of the background'
        },
        triangle_length:{
          type: jsPsych.plugins.parameterType.INT,
          default: 120,
          description: 'length of traingle'
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
        up_key: {
          type: jsPsych.plugins.parameterType.KEYCODE,
          pretty_name: 'Up key',
          default: 'arrowup',
          description: 'The key to be pressed to select the left planet'
        },
        down_key: {
          type: jsPsych.plugins.parameterType.KEYCODE,
          pretty_name: 'Down key',
          default: 'arrowdown',
          description: 'The key to be pressed to select the right planet' 
        },       
//  
        trial_listen_duration: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'trial window duration',
          default: null,
          description: 'How long to show the trial.'
        },      
        choice_listen_duration: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Choice window duration',
          default: null,
          description: 'How long to wait for a response (in milliseconds).'
        },
        iti_duration: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Duration of inter-trial interval',
          default: 1500,
          description: 'How long to display a blank screen between trials (in milliseconds).'
        },

        render_on_canvas: {
          type: jsPsych.plugins.parameterType.BOOL,
          pretty_name: 'Render on canvas',
          default: true,
          description: 'If true, the image will be drawn onto a canvas element (prevents blank screen between consecutive images in some browsers).'+
            'If false, the image will be shown via an img element.'
        }
        
      }
    }
  
    plugin.trial = function(display_element, trial) {
  
      // add a canvas to the HTML_STRING, store its context, and draw a blank background
      var html_str = "";

      html_str += '<p align="center" style="color: green;">*** Submit relative causal rating ***</p>'+'<p align="center" style="color: green;">[&larr;/&rarr;/&uarr;/&darr;] to change, [Y] to confirm.</p>'

      cur_idx = 0
      size_triangle = 110
      draw_space = 30
      //xpos = window.screen.width/2
      //ypos = window.screen.height/2
      
      all_xpos = [...Array(25).keys()]
      all_ypos = [...Array(25).keys()]

      total_size = 3*size_triangle+2*trial.stimulus_dimensions[1];
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
      
      
      // set next circles of each circle
      // [up, left, down, right]

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

      // store response
      var response = {
        rt: null,
        key: null,
        description: null
      };

      function drawstimuli(){
        html_str += '<img src="'+trial.stimuli[0]+'" id="image_0"'+'width="'+trial.stimulus_dimensions[0]+'"height="'+trial.stimulus_dimensions[1]+
        '"style="position:absolute;top:'+(ypos-size_triangle-trial.stimulus_dimensions[1]-draw_space)+'px; left: '+(xpos-(trial.stimulus_dimensions[0]/2)+10)+'px;"></img>'
        html_str += '<img src="'+trial.stimuli[1]+'" id="image_1"'+'width="'+trial.stimulus_dimensions[0]+'"height="'+trial.stimulus_dimensions[1]+
        '"style="position:absolute;top:'+(ypos+size_triangle*Math.cos(Math.PI/3)+draw_space)+'px; left: '+(xpos-size_triangle*Math.sin(Math.PI/3)-trial.stimulus_dimensions[0]-(draw_space-20))+'px;"></img>'
        html_str += '<img src="'+trial.stimuli[2]+'" id="image_2"'+'width="'+trial.stimulus_dimensions[0]+'"height="'+trial.stimulus_dimensions[1]+ 
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

        return [b_value,c_value,a_value] // order: point A, point B, point C
      }

      function showvalues(c_idx){
        values = barycentric_system(c_idx)
        display_element.querySelector('#A').innerHTML = values[0].toFixed(2)
        display_element.querySelector('#B').innerHTML = values[1].toFixed(2)
        display_element.querySelector('#C').innerHTML = values[2].toFixed(2)
      }

      var one_msg_overtime = msg_overtime;
      var startTime = performance.now();
      now_start_time = startTime
      var msg_startTime = null;
      var msg_endTime = null;
      var rating_gap_duration = null;

      
      var end_trial = function(){
        
        // kill any remaining setTimeout handlers
        jsPsych.pluginAPI.clearAllTimeouts();
        clearTimeout(Timeout2);
        jsPsych.pluginAPI.cancelAllKeyboardResponses();
  
        // kill keyboard listeners
        if (typeof keyboardListener !== 'undefined') {
          jsPsych.pluginAPI.cancelKeyboardResponse(keyboardListener);
        }

        jsPsych.pluginAPI.cancelAllKeyboardResponses();

        msg_overtime = one_msg_overtime;
        var endTime = performance.now();
        response.rt = endTime - startTime;
  
        // gather the data to store for the trial

        rating_values = barycentric_system(c_idx);
        
        // save for bonus round
        var rating = [[used_cue_idxes[0],rating_values[0]],[used_cue_idxes[1],rating_values[1]],[used_cue_idxes[2],rating_values[2]]];
        rating.sort(function(a,b){
          return (b[1]-a[1]);
        })

        rating_rounds.push(rating);

        path_0 = (trial.stimuli[0]).split("/")
        path_1 = (trial.stimuli[1]).split("/")
        path_2 = (trial.stimuli[2]).split("/")

        var trial_data = {
          session: parseInt(repetition_count/8),
          round:(repetition_count%8),
          question_timeline: [startTime, endTime],
          msg_timeline: [msg_startTime, msg_endTime],
          rating: [[path_0.pop(),rating_values[0]],[path_1.pop(),rating_values[1]],[path_2.pop(),rating_values[2]]],
          description: response.description
        };
  
        // clear the display
        display_element.innerHTML = '';
  
        jsPsych.finishTrial(trial_data); 
        
      };  

      var normal_response = function(){
        
        // kill any remaining setTimeout handlers
        jsPsych.pluginAPI.clearAllTimeouts();
  
        // kill keyboard listeners
        if (typeof keyboardListener !== 'undefined') {
          jsPsych.pluginAPI.cancelKeyboardResponse(keyboardListener);
        }

        jsPsych.pluginAPI.cancelAllKeyboardResponses();

        var endTime = performance.now();
        response.rt = endTime - now_start_time;

        if (response.rt <= trial.choice_listen_duration){
          response.description = 'correct submit'
        } else {
          response.description = 'late submit'
        }
        
        end_trial();
  
      }

      // initial setting
      
      init_idx = 12;
      c_idx = init_idx;
      init_values = barycentric_system(c_idx)

      html_str +=  "<p id='A' style='color:green;position:absolute;top:"+ (ypos-size_triangle-trial.stimulus_dimensions[1]-draw_space*3)+ "px; left: " 
      + (xpos-(trial.stimulus_dimensions[0]/2)+draw_space*2) +"px'>"+init_values[0].toFixed(2)+"</p>"; // point A
      html_str +=  "<p id='B' style='color:green;position:absolute;top:"+ (ypos+size_triangle*Math.cos(Math.PI/3)+trial.stimulus_dimensions[1]+draw_space)+ "px; left: " 
      + (xpos-size_triangle*Math.sin(Math.PI/3)-trial.stimulus_dimensions[0]+draw_space*1.5) +"px'>"+init_values[1].toFixed(2)+"</p>"; // point B
      html_str +=  "<p id='C' style='color:green;position:absolute;top:"+ (ypos+size_triangle*Math.cos(Math.PI/3)+trial.stimulus_dimensions[1]+draw_space)+ "px; left: " 
      + (xpos+size_triangle*Math.sin(Math.PI/3)+draw_space*2.7) +"px'>"+init_values[2].toFixed(2)+"</p>"; //point C

      
      // callback function of keyboardlistener
      var after_response = function(info) {

        //console.log(c_idx)
        next_idxs = next_circle[c_idx]
        createTimeout2(trial.trial_listen_duration);
        createkeyboardListener();

        if (info.key === trial.left_key) {
          
          //console.log('left')
          new_idx = next_idxs[1]
          if (new_idx !=0) {
            highlight_circle(new_idx,c_idx)
            showvalues(new_idx)
            c_idx = new_idx 
          }

        } else if (info.key === trial.right_key) {
          
          //console.log('right')
          new_idx = next_idxs[3]
          if (new_idx !=0) {
            highlight_circle(new_idx,c_idx)
            showvalues(new_idx)
            c_idx = new_idx 
          }

        } else if (info.key === trial.up_key) {
          //console.log('up')
          new_idx = next_idxs[0]
          if (new_idx !=0) {
            highlight_circle(new_idx,c_idx)
            showvalues(new_idx)
            c_idx = new_idx 
          }
          
        } else if (info.key === trial.down_key) {
          //console.log('down')
          new_idx = next_idxs[2]
          if (new_idx !=0) {
            highlight_circle(new_idx,c_idx)
            showvalues(new_idx)
            c_idx = new_idx 
          }

        } else if (info.key === 'y'){
          //console.log('y')
          normal_response();
        }
      };      


    var createkeyboardListener = function() {
        var keyboardListener = jsPsych.pluginAPI.getKeyboardResponse({
          callback_function: after_response,
          valid_responses: [trial.left_key, trial.right_key, trial.up_key,trial.down_key, 'y'],
          rt_method: 'performance',
          persist: false,
          allow_held_key: false,
        });   
      }
 
    // timeout for submit time limit after one response
    var createTimeout1 = function(msg_time) {
        if (trial.trial_listen_duration !== null) {
          jsPsych.pluginAPI.setTimeout(function() {
            response.description = 'no submit';
            end_trial();
            //missed_response();
          }, trial.trial_listen_duration-msg_time);
        }
      }

    // timeout for response time limit
    var createTimeout2 = function(trial_duration) {
      clearTimeout(Timeout2);
      Timeout2 = window.setTimeout(timeout_response, trial_duration);
    }

    // show msg over two rating
    var over_timeout_response = function() {

      //console.log("over")
      //console.log(msg_overtime)

      // Kill all setTimeout handlers.
      jsPsych.pluginAPI.clearAllTimeouts();
      jsPsych.pluginAPI.cancelAllKeyboardResponses();
      //clearTimeout(Timeout2);

      // Display warning message.
      if (msg_count < chance) {
        var msg = '<p id = "msg" style="font-size: 20px; line-height: 1.5em">You did not respond within the allotted time. Please pay more attention on the next trial.<br><br><b>Warning:</b> If you miss too many trials, we may end the experiment early and reject your work.';
      }
      else {
        var msg = '<p id = "msg" style="font-size: 20px; line-height: 1.5em">we end the experiment early and reject your work</p>';
      }
      
      display_element.innerHTML = msg;


      if (msg_count >= chance) {
        low_quality = true;
        msg_timeout = window.setTimeout(function() {
          jsPsych.endExperiment();
         }, msg_duration); 
      } else {
        msg_timeout = window.setTimeout(function() {
          msg_endTime = performance.now();
          html_gap = "<p style='font-familiy:IBM Plex Sans Extrabold;font-size:24px;'>+</p>";
          display_element.innerHTML = html_gap;
          rating_gap_duration = Math.floor(Math.random()*(100)+200);
          
          jsPsych.pluginAPI.setTimeout(function() {
            display_element.innerHTML = html_str;
            display_element.querySelector('#C'+init_idx).className = 'plus_selected';
            createTimeout2(trial.trial_listen_duration);
            createkeyboardListener();
            createTimeout1(msg_overtime);
            clearTimeout(msg_timeout);
          }, (rating_gap_duration));

          }, msg_overtime);  
      }

      one_msg_overtime = 0;
    }

      // no response past the time limit 
      var timeout_response = function() {

        //console.log("timeout")

        // Kill all setTimeout handlers.
        jsPsych.pluginAPI.clearAllTimeouts();
        jsPsych.pluginAPI.cancelAllKeyboardResponses();
        clearTimeout(Timeout2);

        msg_count += 1;

        // Display warning message.
        if (msg_count < chance) {
          var msg = '<p id = "msg" style="font-size: 20px; line-height: 1.5em">You did not respond within the allotted time. Please pay more attention on the next trial.<br><br><b>Warning:</b> If you miss too many trials, we may end the experiment early and reject your work.';
        }
        else {
          var msg = '<p id = "msg" style="font-size: 20px; line-height: 1.5em">we end the experiment early and reject your work</p>';
        }


        display_element.innerHTML = msg;

        var curr_time = performance.now();
        var remain_time = (trial.trial_listen_duration - (curr_time - now_start_time));
        msg_startTime = performance.now();

        //console.log("real start " + now_start_time);
        //console.log("remain " + remain_time);

        if (msg_count >= chance) {
          low_quality = true;
          msg_timeout = window.setTimeout(function() {
            jsPsych.endExperiment();
           }, msg_duration); 
        } else {
          if (remain_time > msg_duration){  
            one_msg_overtime = 0;
            msg_timeout = window.setTimeout(function() {
                msg_endTime = performance.now();
                display_element.innerHTML = html_str;
                display_element.querySelector('#C'+init_idx).className = 'plus_selected';
                createTimeout2(trial.trial_listen_duration);
                createkeyboardListener();
                createTimeout1(trial.trial_listen_duration-(remain_time-msg_duration));
                clearTimeout(msg_timeout);
            }, msg_duration); 
          } else {
            one_msg_overtime = 0;
            msg_timeout = window.setTimeout(function() {
             // display_element.innerHTML = html_str;
             // display_element.querySelector('#C'+init_idx).className = 'plus_selected'
             // createkeyboardListener();
              clearTimeout(msg_timeout);
              response.description = 'no submit';
              end_trial();
            }, msg_duration); 
          }
        }
      }          
      

      html_str += '</div>';

      // initial setting
      display_element.innerHTML = html_str;

      display_element.querySelector('#C'+init_idx).className = 'plus_selected'

      // change timeout2 because timeout response of tri rating and of l/c rating are not same
      clearTimeout(Timeout2);
      var remain_timeout2_time = timeout_duration-((performance.now())-timeout2_start);
      //console.log("remain timeout2 " + remain_timeout2_time);
      createTimeout2(remain_timeout2_time);

      createkeyboardListener();

      if (msg_overtime == 0){
        createTimeout1(msg_overtime);
      } else {
        over_timeout_response();
      }
      
      
    //  display_element.getElementById('C0').focus();
  
    } // end plugin.trial
  
    return plugin;
  
  })(); // end plugin function


