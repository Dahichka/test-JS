var greeting = "Hello ";
var name = "Dania";
console.log(greeting + name);
console.log((greeting + name).length);
console.log(name[0]);
console.log(name.slice(1));
console.log(("I am " + name).toUpperCase());
var randomWords = ["Planet", "Worm", "Flower", "Competer"];
var pickRandomWord = function (words){
return words[Math.floor(Math.random()*randomWords.length)];}
console.log(pickRandomWord(randomWords));