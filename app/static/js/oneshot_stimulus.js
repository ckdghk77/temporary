// specify the images corresponding to simple cues

var cue_images = []

for (idx=1; idx<=120; idx++)
{

  i_path = "../static/img/cue_imgs/" + ("00" + idx).slice(-3) + ".png"
  cue_images.push(i_path)
}

var tuto_images = []
for (idx=1; idx<=3; idx++)
{
  i_path = "../static/img/cue_imgs/test0" + idx + ".png"
  tuto_images.push(i_path)
}


var reward_images_name = ['-10','-50','10','50']

 var reward_images = [
    '../static/img/reward_imgs/msg_n10pt.PNG',
    '../static/img/reward_imgs/msg_n50pt.PNG',
    '../static/img/reward_imgs/msg_p10pt.PNG',
    '../static/img/reward_imgs/msg_p50pt.PNG',
 ]


 var inst_images = []

 for (idx=1;idx<=37;idx++){

   inst_path =  "../static/img/Inst_imgs/inst" + idx + ".PNG"
   inst_images.push(inst_path)
 }
  

 function alloc_cues()
 {
    cue_infos = {}
 }

 function alloc_rewards()
 {
    reward_infos = {}

 }

 var unused_cue_idxes = [...Array(cue_images.length).keys()];

 function set_rounds_type()
 {  
    rounds_type = {}

    // os_type: 0 = oneshot, 1 = incremental
    // n_or_p : 0 = n = (non-novel = +10, novel = -50), 1 = p = (non-novel = -10, novel = +50)
    
    os_type = [0,0,0,0,1,1,1,1]
    n_or_p = [0,0,0,0,1,1,1,1]

    os_type = jsPsych.randomization.shuffle(os_type)
    n_or_p = jsPsych.randomization.shuffle(n_or_p)

    rounds_type.os_type = os_type
    rounds_type.n_or_p = n_or_p

    return rounds_type    
 }


 function alloc_all(os_type=0, n_or_p=0, driving=0, verbose=false) // select images and rewards for one trial
 {
    total_infos = {}
    
    if(n_or_p == 0){ // n
      total_infos.rew_idxes = [2,1] // [non-novel, novel]   
      novel_outcome = 1
      non_novel_outcome = 2 
    }
    else{ //p
      total_infos.rew_idxes = [0,3]
      novel_outcome = 3
      non_novel_outcome = 0    
    }

    cue_idxes = jsPsych.randomization.sampleWithoutReplacement(unused_cue_idxes, 3) // randomly sample 3 images
    //console.log("unused_cue_idxes : " + unused_cue_idxes.length)
    total_infos.cue_idxes = cue_idxes
  
    if(driving==0){
    // allocate frequency in which images are presented
    cue_allocation = [...Array(25).keys()]
    cue_allocation = cue_allocation.fill(cue_idxes[0],0,16)
    cue_allocation = cue_allocation.fill(cue_idxes[1],16,24)
    cue_allocation = cue_allocation.fill(cue_idxes[2],24)
    cue_allocation = jsPsych.randomization.shuffle(cue_allocation) // shuffle images order

    total_infos.cue_allocation = cue_allocation
    //console.log(cue_allocation)
  
    if (os_type == 0 ){ // oneshot
      reward_allocation = Array(5).fill(non_novel_outcome)
      reward_allocation[parseInt(cue_allocation.indexOf(cue_idxes[2])/5)] = novel_outcome
    } else { //1 // incremental
      reward_allocation = Array(5).fill(non_novel_outcome)
      novel_cue_idx = parseInt(cue_allocation.indexOf(cue_idxes[2])/5)
      nonnovel_cue_idxs = [0,1,2,3,4]
      nonnovel_cue_idxs.splice(nonnovel_cue_idxs.indexOf(novel_cue_idx),1)
      
      novel_reward_idx = jsPsych.randomization.sampleWithoutReplacement(nonnovel_cue_idxs, 1)
      reward_allocation[novel_reward_idx] = novel_outcome
    }

    //print(reward_allocation)
    total_infos.reward_allocation = reward_allocation
    }else{
      
      $.ajax({
        type: "GET",
        url: `/gen_exps?ostype=${os_type}`,
        async: false,
        //data : {os_type:os_type},
        success: function(response){
          obj = JSON.parse(response)
          cue_allocs = obj.cue_alloc;
          rew_allocs = obj.rew_alloc;

          for (let i = 0; i < cue_allocs.length; i++) {
            if (cue_allocs[i] == "0")
            {
                cue_allocs[i] = cue_idxes[0]
            }else if (cue_allocs[i] == "1")
            {
                cue_allocs[i] = cue_idxes[1]
            }else if (cue_allocs[i] == "2")
            {
                cue_allocs[i] = cue_idxes[2]
            }
          }
          
          for (let i = 0; i < rew_allocs.length; i++) {
            if (rew_allocs[i] == "3")
            {
                rew_allocs[i] = non_novel_outcome
            }else if (rew_allocs[i] == "4")
            {
                rew_allocs[i] = novel_outcome
            }
          }

          total_infos.cue_allocation = cue_allocs
          total_infos.reward_allocation = rew_allocs
          
        }
      }
      )
      
    }

    return total_infos
 }
 