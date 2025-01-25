const results = require('./results.json')
const train_section_data2 = require('./train_section_data2.json')

const differences = results.filter(e => e.selected_checkbox != train_section_data2[e.image_index]?.question_number)
console.log({
    none: differences.filter(e=> e.status == 'None').length,
    none_ids: differences.filter(e=> e.status == 'None').map(e=> e.image_index),
    incorrect: differences.filter(e=> e.status == 'Selected').length,
    incorrect_ids:differences.filter(e=> e.status == 'Selected').map(e=> [e.image_index,e.selected_checkbox]),   

}, {
    total: results.length, incorrect: differences.length
})