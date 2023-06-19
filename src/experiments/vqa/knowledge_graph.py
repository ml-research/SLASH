"""
The source code is based on:
Scallop: From Probabilistic Deductive Databases to Scalable Differentiable Reasoning
Jiani Huang, Ziyang Li, Binghong Chen, Karan Samel, Mayur Naik, Le Song, Xujie Si
Advances in Neural Information Processing Systems 34 (NeurIPS 2021)
https://proceedings.neurips.cc/paper/2021/hash/d367eef13f90793bd8121e2f675f0dc2-Abstract.html
"""

import re 

KG_FACTS = '''
%FACTS
is_a("boat", "watercraft").
is_a("photograph", "artwork").
is_a("chalk", "writing implement").
is_a("tv", "electrical appliance").
is_a("screen", "electronic device").
is_a("chocolate", "dessert").
is_a("pig", "even-toed ungulate").
is_a("pig", "mammal").
is_a("alcohol", "alcoholic beverage").
is_a("wine", "alcoholic beverage").
is_a("picture", "artwork").
is_a("zebra", "herbivore").
is_a("skillet", "kitchen utensil").
is_a("horse", "mammal").
is_a("rice", "staple food").
is_a("rice", "grains").
is_a("zebra", "mammal").
is_a("refrigerator", "electrical appliance").
is_a("soda", "soft drinks").
is_a("horse", "herbivore").
is_a("air conditioner", "electrical appliance").
is_a("container", "kitchen utensil").
is_a("cat", "feline").
is_a("cat", "mammal").
is_a("zebra", "odd-toed ungulate").
is_a("spoon", "eating utensil").
is_a("lemon", "citrus fruit").
is_a("lime", "citrus fruit").
is_a("soap", "toiletry").
is_a("dog", "mammal").
is_a("dog", "carnivora").
is_a("giraffe", "even-toed ungulate").
is_a("giraffe", "mammal").
is_a("giraffe", "herbivore").
is_a("elephant", "mammal").
is_a("bear", "carnivora").
is_a("duck", "bird").
is_a("sheep", "mammal").
is_a("sheep", "even-toed ungulate").
is_a("computer", "electronic device").
is_a("fork", "eating utensil").
is_a("pen", "writing implement").
is_a("speaker", "electronic device").
is_a("camera", "electronic device").
is_a("bus", "public transports").
is_a("tiger", "feline").
is_a("painting", "artwork").
is_a("headphones", "electronic device").
is_a("ostrich", "bird").
is_a("projector", "electronic device").
is_a("mousepad", "electronic device").
is_a("monitor", "electronic device").
is_a("orange", "citrus fruit").
is_a("keyboard", "electronic device").
is_a("airplane", "public transports").
is_a("airplane", "aircraft").
is_a("pot", "kitchen utensil").
is_a("microwave", "electrical appliance").
is_a("bird", "class").
is_a("fox", "canidae").
is_a("oven", "electrical appliance").
is_a("kettle", "kitchen utensil").
is_a("tea pot", "kitchen utensil").
is_a("washing machine", "electrical appliance").
is_a("donkey", "mammal").
is_a("macaroni", "main course").
is_a("propeller", "aircraft").
is_a("sink", "kitchen utensil").
is_a("landscape", "artwork").
is_a("blade", "kitchen utensil").
is_a("marker", "writing implement").
is_a("tv", "electronic device").
is_a("radio", "electronic device").
is_a("printer", "electronic device").
is_a("horse", "odd-toed ungulate").
is_a("train", "public transports").
is_a("sheep", "herbivore").
is_a("knife", "eating utensil").
is_a("wii", "electronic device").
is_a("guitar", "instruments").
is_a("bull", "mammal").
is_a("pan", "kitchen utensil").
is_a("grapefruit", "citrus fruit").
is_a("lizard", "reptile").
is_a("jet", "aircraft").
is_a("mouse", "electronic device").
is_a("toothbrush", "toiletry").
is_a("lion", "feline").
is_a("lion", "carnivora").
is_a("tea kettle", "kitchen utensil").
is_a("cake", "dessert").
is_a("laptop", "electronic device").
is_a("cow", "mammal").
is_a("calf", "mammal").
is_a("phone", "electronic device").
is_a("pencil", "writing implement").
is_a("game controller", "electronic device").
is_a("pizza", "main course").
is_a("pizza", "fast food").
is_a("calculator", "electronic device").
is_a("ice cream", "dessert").
is_a("cookie", "dessert").
is_a("grill", "kitchen utensil").
is_a("goat", "mammal").
is_a("microphone", "electronic device").
is_a("polar bear", "mammal").
is_a("statue", "artwork").
is_a("cow", "even-toed ungulate").
is_a("chopsticks", "eating utensil").
is_a("cow", "herbivore").
is_a("cattle", "even-toed ungulate").
is_a("dishwasher", "electrical appliance").
is_a("blender", "kitchen utensil").
is_a("coffee pot", "kitchen utensil").
is_a("cooking pot", "kitchen utensil").
is_a("coffee maker", "electrical appliance").
is_a("water", "liquid").
is_a("dvd", "electronic device").
is_a("ship", "watercraft").
is_a("fax machine", "electronic device").
is_a("lotion", "toiletry").
is_a("pepper", "condiment").
is_a("noodles", "main course").
is_a("bread", "staple food").
is_a("cereal", "grains").
is_a("drum", "instruments").
is_a("graffiti", "artwork").
is_a("deer", "herbivore").
is_a("deer", "even-toed ungulate").
is_a("cutting board", "kitchen utensil").
is_a("knife block", "kitchen utensil").
is_a("whisk", "kitchen utensil").
is_a("burger", "fast food").
is_a("burger", "main course").
is_a("heater", "electrical appliance").
is_a("rolling pin", "kitchen utensil").
is_a("dog", "canidae").
is_a("cattle", "mammal").
is_a("harp", "instruments").
is_a("cat", "carnivora").
is_a("pigeon", "bird").
is_a("juicer", "kitchen utensil").
is_a("subway", "public transports").
is_a("tiger", "mammal").
is_a("penguin", "bird").
is_a("pizza oven", "electrical appliance").
is_a("pizza cutter", "kitchen utensil").
is_a("bear", "mammal").
is_a("seagull", "bird").
is_a("ladle", "kitchen utensil").
is_a("foil", "kitchen utensil").
is_a("wii controller", "electronic device").
is_a("router", "electronic device").
is_a("panda", "mammal").
is_a("monkey", "mammal").
is_a("spatula", "kitchen utensil").
is_a("rabbit", "herbivore").
is_a("bunny", "mammal").
is_a("mixing bowl", "kitchen utensil").
is_a("cola", "soft drinks").
is_a("turtle", "reptile").
is_a("frying pan", "kitchen utensil").
is_a("cheesecake", "dessert").
is_a("baking sheet", "kitchen utensil").
is_a("dryer", "electrical appliance").
is_a("beer", "alcoholic beverage").
is_a("calf", "herbivore").
is_a("sailboat", "watercraft").
is_a("toaster oven", "electrical appliance").
is_a("bull", "herbivore").
is_a("lion", "mammal").
is_a("dvd player", "electronic device").
is_a("toothpick", "eating utensil").
is_a("squirrel", "mammal").
is_a("squirrel", "rodent").
is_a("tongs", "kitchen utensil").
is_a("moose", "even-toed ungulate").
is_a("moose", "mammal").
is_a("pasta", "main course").
is_a("pony", "mammal").
is_a("sword", "weapon").
is_a("fox", "mammal").
is_a("toaster", "electrical appliance").
is_a("eagle", "bird").
is_a("lipstick", "cosmetic").
is_a("calf", "even-toed ungulate").
is_a("rifle", "weapon").
is_a("piano", "instruments").
is_a("cupcake", "dessert").
is_a("feline", "family").
is_a("chicken", "bird").
is_a("pie", "dessert").
is_a("dish drainer", "kitchen utensil").
is_a("lamb", "even-toed ungulate").
is_a("lamb", "mammal").
is_a("goat", "even-toed ungulate").
is_a("goat", "herbivore").
is_a("lamb", "herbivore").
is_a("puppy", "canidae").
is_a("basil", "condiment").
is_a("fries", "fast food").
is_a("grater", "kitchen utensil").
is_a("sauce", "condiment").
is_a("kitten", "feline").
is_a("hippo", "even-toed ungulate").
is_a("hippo", "mammal").
is_a("bass", "instruments").
is_a("antelope", "herbivore").
is_a("polar bear", "carnivora").
is_a("corn", "grains").
is_a("corn", "staple food").
is_a("brownie", "dessert").
is_a("cash", "currency").
is_a("cereal", "staple food").
is_a("owl", "bird").
is_a("flamingo", "bird").
is_a("monkey", "primate").
is_a("bomb", "weapon").
is_a("pizza pan", "kitchen utensil").
is_a("bull", "even-toed ungulate").
is_a("rice cooker", "electrical appliance").
is_a("sushi", "main course").
is_a("vacuum", "electrical appliance").
is_a("toothpaste", "toiletry").
is_a("frog", "amphibian").
is_a("antelope", "mammal").
is_a("hen", "bird").
is_a("puppy", "mammal").
is_a("puppy", "carnivora").
is_a("double decker", "public transports").
is_a("alligator", "reptile").
is_a("deer", "mammal").
is_a("cinnamon", "condiment").
is_a("tongs", "eating utensil").
is_a("kitten", "carnivora").
is_a("kitten", "mammal").
is_a("cattle", "herbivore").
is_a("antelope", "even-toed ungulate").
is_a("parrot", "bird").
is_a("ipod", "electronic device").
is_a("dolphin", "even-toed ungulate").
is_a("leopard", "mammal").
is_a("fox", "carnivora").
is_a("gun", "weapon").
is_a("canoe", "watercraft").
is_a("food processor", "electrical appliance").
is_a("hamburger", "main course").
is_a("hamburger", "fast food").
is_a("console", "electronic device").
is_a("xbox controller", "electronic device").
is_a("video camera", "electronic device").
is_a("mammal", "class").
is_a("helicopter", "aircraft").
is_a("spear", "weapon").
is_a("hummingbird", "bird").
is_a("goose", "bird").
is_a("rodent", "order").
is_a("shampoo", "toiletry").
is_a("violin", "instruments").
is_a("coin", "currency").
is_a("dove", "bird").
is_a("champagne", "alcoholic beverage").
is_a("camel", "even-toed ungulate").
is_a("bison", "even-toed ungulate").
is_a("bison", "herbivore").
is_a("trumpet", "instruments").
is_a("rabbit", "mammal").
is_a("kangaroo", "mammal").
is_a("wolf", "carnivora").
is_a("guacamole", "condiment").
is_a("wok", "kitchen utensil").
is_a("tiger", "carnivora").
is_a("panda bear", "mammal").
is_a("syrup", "condiment").
is_a("moose", "herbivore").
is_a("hawk", "bird").
is_a("beaver", "mammal").
is_a("liquor", "alcoholic beverage").
is_a("shaving cream", "toiletry").
is_a("utensil holder", "kitchen utensil").
is_a("cake pan", "kitchen utensil").
is_a("seal", "mammal").
is_a("seal", "carnivora").
is_a("poodle", "mammal").
is_a("raccoon", "carnivora").
is_a("ice maker", "electrical appliance").
is_a("hamster", "rodent").
is_a("rat", "mammal").
is_a("crow", "bird").
is_a("swan", "bird").
is_a("whale", "even-toed ungulate").
is_a("garlic", "condiment").
is_a("wolf", "mammal").
is_a("mustard", "condiment").
is_a("deodorant", "toiletry").
is_a("hand dryer", "electrical appliance").
is_a("wheat", "staple food").
is_a("wheat", "grains").
is_a("pudding", "dessert").
is_a("accordion", "instruments").
is_a("peacock", "bird").
is_a("crayon", "writing implement").
is_a("gorilla", "primate").
is_a("cheetah", "carnivora").
is_a("cheetah", "mammal").
is_a("leopard", "feline").
is_a("baking pan", "kitchen utensil").
is_a("mascara", "cosmetic").
is_a("gorilla", "mammal").
is_a("fingernail polish", "cosmetic").
is_a("rhino", "odd-toed ungulate").
is_a("hamster", "mammal").
is_a("rat", "rodent").
is_a("sugar", "condiment").
is_a("finch", "bird").
is_a("skewer", "eating utensil").
is_a("cheeseburger", "main course").
is_a("snake", "reptile").
is_a("bald eagle", "bird").
is_a("cheetah", "feline").
is_a("bison", "mammal").
is_a("rhino", "mammal").
is_a("camel", "herbivore").
is_a("camel", "mammal").
is_a("ketchup", "condiment").
is_a("ape", "primate").
is_a("ape", "mammal").
is_a("gravy", "condiment").
is_a("barley", "grains").
is_a("butter", "condiment").
is_a("copier", "electronic device").
is_a("robin", "bird").
is_a("reptile", "class").
is_a("beaver", "rodent").
is_a("perfume", "cosmetic").
is_a("eyeshadow", "cosmetic").
is_a("wolf", "canidae").
is_a("badger", "mammal").
is_a("leopard", "carnivora").
is_a("eyeliner", "cosmetic").
is_a("people", "mammal").
is_a("salt", "condiment").
is_a("grey cat", "mammal").
is_a("alpaca", "even-toed ungulate").
is_a("raccoon", "mammal").
is_a("perfume", "toiletry").
is_a("caramel", "condiment").
is_a("clarinet", "instruments").
is_a("otter", "carnivora").
is_a("athlete", "person").
is_a("occupation", "person").
is_a("baseball position", "person").
is_a("tennis player", "athlete").
is_a("baseball player", "athlete").
is_a("soccer player", "athlete").
is_a("basketball player", "athlete").
is_a("frisbee player", "athlete").
is_a("football player", "athlete").
is_a("volleyball player", "athlete").
is_a("billiards player", "athlete").
is_a("hockey player", "athlete").
is_a("golfer", "athlete").
is_a("surfer", "athlete").
is_a("biker", "athlete").
is_a("swimmer", "athlete").
is_a("runner", "athlete").
is_a("jogger", "athlete").
is_a("skier", "athlete").
is_a("skateboarder", "athlete").
is_a("skater", "athlete").
is_a("snowboarder", "athlete").
is_a("police", "occupation").
is_a("teacher", "occupation").
is_a("student", "occupation").
is_a("pilot", "occupation").
is_a("cowboy", "occupation").
is_a("soldier", "occupation").
is_a("fisherman", "occupation").
is_a("worker", "occupation").
is_a("photographer", "occupation").
is_a("performer", "occupation").
is_a("farmer", "occupation").
is_a("policeman", "occupation").
is_a("officer", "occupation").
is_a("vendor", "occupation").
is_a("shopper", "occupation").
is_a("bus driver", "occupation").
is_a("driver", "occupation").
is_a("jockey", "occupation").
is_a("engineer", "occupation").
is_a("doctor", "occupation").
is_a("chef", "occupation").
is_a("baker", "occupation").
is_a("bartender", "occupation").
is_a("waiter", "occupation").
is_a("waitress", "occupation").
is_a("customer", "occupation").
is_a("player", "occupation").
is_a("athlete", "occupation").
is_a("coach", "occupation").
is_a("actor", "occupation").
is_a("batter", "baseball position").
is_a("catcher", "baseball position").
is_a("pitcher", "baseball position").
is_a("umpire", "baseball position").
is_a("santa", "character").
is_a("mickey mouse", "character").
is_a("snoopy", "character").
is_a("pikachu", "character").
is_a("leg", "part of body").
is_a("tail", "part of body").
is_a("lap", "part of body").
is_a("neck", "part of body").
is_a("foot", "part of body").
is_a("face", "part of body").
is_a("arm", "part of body").
is_a("hand", "part of body").
is_a("wrist", "part of body").
is_a("shoulder", "part of body").
is_a("head", "part of body").
is_a("horn", "part of body").
is_a("tusk", "part of body").
is_a("racing", "sport").
is_a("baseball", "sport").
is_a("soccer", "sport").
is_a("skiing", "sport").
is_a("basketball", "sport").
is_a("polo", "sport").
is_a("tennis", "sport").
is_a("surfing", "sport").
is_a("riding", "sport").
is_a("skateboarding", "sport").
is_a("skate", "sport").
is_a("swimming", "sport").
is_a("snowboarding", "sport").
is_a("christmas", "event").
is_a("thanksgiving", "event").
is_a("wedding", "event").
is_a("birthday", "event").
is_a("halloween", "event").
is_a("party", "event").
is_a("cat", "animal").
is_a("kitten", "animal").
is_a("dog", "animal").
is_a("puppy", "animal").
is_a("poodle", "animal").
is_a("bull", "animal").
is_a("cow", "animal").
is_a("cattle", "animal").
is_a("bison", "animal").
is_a("calf", "animal").
is_a("pig", "animal").
is_a("ape", "animal").
is_a("monkey", "animal").
is_a("gorilla", "animal").
is_a("rat", "animal").
is_a("squirrel", "animal").
is_a("hamster", "animal").
is_a("deer", "animal").
is_a("moose", "animal").
is_a("alpaca", "animal").
is_a("elephant", "animal").
is_a("goat", "animal").
is_a("sheep", "animal").
is_a("lamb", "animal").
is_a("antelope", "animal").
is_a("rhino", "animal").
is_a("hippo", "animal").
is_a("giraffe", "animal").
is_a("zebra", "animal").
is_a("horse", "animal").
is_a("pony", "animal").
is_a("donkey", "animal").
is_a("camel", "animal").
is_a("panda", "animal").
is_a("panda bear", "animal").
is_a("bear", "animal").
is_a("polar bear", "animal").
is_a("seal", "animal").
is_a("fox", "animal").
is_a("raccoon", "animal").
is_a("tiger", "animal").
is_a("wolf", "animal").
is_a("lion", "animal").
is_a("leopard", "animal").
is_a("cheetah", "animal").
is_a("badger", "animal").
is_a("rabbit", "animal").
is_a("bunny", "animal").
is_a("beaver", "animal").
is_a("kangaroo", "animal").
is_a("dinosaur", "animal").
is_a("dragon", "animal").
is_a("fish", "animal").
is_a("whale", "animal").
is_a("dolphin", "animal").
is_a("crab", "animal").
is_a("shark", "animal").
is_a("octopus", "animal").
is_a("lobster", "animal").
is_a("oyster", "animal").
is_a("butterfly", "animal").
is_a("bee", "animal").
is_a("fly", "animal").
is_a("ant", "animal").
is_a("firefly", "animal").
is_a("snail", "animal").
is_a("spider", "animal").
is_a("bird", "animal").
is_a("penguin", "animal").
is_a("pigeon", "animal").
is_a("seagull", "animal").
is_a("finch", "animal").
is_a("robin", "animal").
is_a("ostrich", "animal").
is_a("goose", "animal").
is_a("owl", "animal").
is_a("duck", "animal").
is_a("hawk", "animal").
is_a("eagle", "animal").
is_a("swan", "animal").
is_a("chicken", "animal").
is_a("hen", "animal").
is_a("hummingbird", "animal").
is_a("parrot", "animal").
is_a("crow", "animal").
is_a("flamingo", "animal").
is_a("peacock", "animal").
is_a("bald eagle", "animal").
is_a("dove", "animal").
is_a("snake", "animal").
is_a("lizard", "animal").
is_a("alligator", "animal").
is_a("turtle", "animal").
is_a("frog", "animal").
is_a("butterfly", "insect").
is_a("bee", "insect").
is_a("fly", "insect").
is_a("ant", "insect").
is_a("firefly", "insect").
is_a("swan", "aquatic bird").
is_a("penguin", "aquatic bird").
is_a("duck", "aquatic bird").
is_a("goose", "aquatic bird").
is_a("seagull", "aquatic bird").
is_a("flamingo", "aquatic bird").
is_a("tree", "plant").
is_a("flower", "plant").
is_a("bamboo", "tree").
is_a("palm tree", "tree").
is_a("pine", "tree").
is_a("pine tree", "tree").
is_a("oak tree", "tree").
is_a("christmas tree", "tree").
is_a("flower", "flower").
is_a("sunflower", "flower").
is_a("daisy", "flower").
is_a("orchid", "flower").
is_a("seaweed", "flower").
is_a("blossom", "flower").
is_a("lily", "flower").
is_a("rose", "flower").
is_a("sweater", "tops").
is_a("pullover", "tops").
is_a("blouse", "tops").
is_a("blazer", "tops").
is_a("cardigan", "tops").
is_a("halter", "tops").
is_a("parka", "tops").
is_a("turtleneck", "tops").
is_a("hoodie", "tops").
is_a("bikini", "tops").
is_a("tank top", "tops").
is_a("vest", "tops").
is_a("jersey", "tops").
is_a("t shirt", "tops").
is_a("polo shirt", "tops").
is_a("dress shirt", "tops").
is_a("undershirt", "tops").
is_a("shirt", "tops").
is_a("sweatshirt", "tops").
is_a("jacket", "tops").
is_a("jeans", "pants").
is_a("khaki", "pants").
is_a("denim", "pants").
is_a("jogger", "pants").
is_a("capris", "pants").
is_a("trunks", "pants").
is_a("leggings", "pants").
is_a("trousers", "pants").
is_a("shorts", "pants").
is_a("snow pants", "pants").
is_a("shoe", "shoes").
is_a("heel", "shoes").
is_a("tennis shoe", "shoes").
is_a("boot", "shoes").
is_a("sneaker", "shoes").
is_a("sandal", "shoes").
is_a("cleat", "shoes").
is_a("slipper", "shoes").
is_a("flip flop", "shoes").
is_a("flipper", "shoes").
is_a("nightdress", "dress").
is_a("kimono", "dress").
is_a("onesie", "dress").
is_a("sundress", "dress").
is_a("wedding dress", "dress").
is_a("strapless", "dress").
is_a("pants", "clothing").
is_a("tops", "clothing").
is_a("dress", "clothing").
is_a("skirt", "clothing").
is_a("coat", "clothing").
is_a("suit", "clothing").
is_a("jumpsuit", "clothing").
is_a("gown", "clothing").
is_a("robe", "clothing").
is_a("bathrobe", "clothing").
is_a("socks", "clothing").
is_a("helmet", "hat").
is_a("cap", "hat").
is_a("beanie", "hat").
is_a("visor", "hat").
is_a("hood", "hat").
is_a("bandana", "hat").
is_a("baseball cap", "hat").
is_a("headband", "hat").
is_a("cowboy hat", "hat").
is_a("chef hat", "hat").
is_a("mitt", "glove").
is_a("baseball glove", "glove").
is_a("baseball mitt", "glove").
is_a("mitten", "glove").
is_a("necklace", "jewelry").
is_a("ring", "jewelry").
is_a("earring", "jewelry").
is_a("bracelet", "jewelry").
is_a("shoes", "accessory").
is_a("scarf", "accessory").
is_a("tie", "accessory").
is_a("veil", "accessory").
is_a("mask", "accessory").
is_a("apron", "accessory").
is_a("poncho", "accessory").
is_a("cape", "accessory").
is_a("belt", "accessory").
is_a("handkerchief", "accessory").
is_a("hairband", "accessory").
is_a("life jacket", "accessory").
is_a("lanyard", "accessory").
is_a("crown", "accessory").
is_a("garland", "accessory").
is_a("wristband", "accessory").
is_a("watch", "accessory").
is_a("wristwatch", "accessory").
is_a("pocket watch", "accessory").
is_a("hat", "accessory").
is_a("glove", "accessory").
is_a("jewelry", "accessory").
is_a("glasses", "accessory").
is_a("eye glasses", "glasses").
is_a("sunglasses", "glasses").
is_a("goggles", "glasses").
is_a("shrimp", "meat").
is_a("ribs", "meat").
is_a("steak", "meat").
is_a("beef", "meat").
is_a("egg", "meat").
is_a("egg shell", "meat").
is_a("chicken", "meat").
is_a("chicken breast", "meat").
is_a("pepperoni", "meat").
is_a("bacon", "meat").
is_a("ham", "meat").
is_a("sausage", "meat").
is_a("pork", "meat").
is_a("bacon", "pork").
is_a("ham", "pork").
is_a("pepperoni", "sausage").
is_a("salami", "sausage").
is_a("apple", "fruit").
is_a("pineapple", "fruit").
is_a("banana", "fruit").
is_a("olives", "fruit").
is_a("orange", "fruit").
is_a("grapes", "fruit").
is_a("strawberry", "fruit").
is_a("cherry", "fruit").
is_a("lemon", "fruit").
is_a("lime", "fruit").
is_a("mango", "fruit").
is_a("peach", "fruit").
is_a("tangerine", "fruit").
is_a("grape", "fruit").
is_a("kiwi", "fruit").
is_a("pear", "fruit").
is_a("watermelon", "fruit").
is_a("berry", "fruit").
is_a("blueberry", "fruit").
is_a("raspberry", "fruit").
is_a("cranberry", "fruit").
is_a("raisin", "fruit").
is_a("gourd", "fruit").
is_a("grapefruit", "fruit").
is_a("melon", "fruit").
is_a("pomegranate", "fruit").
is_a("papaya", "fruit").
is_a("coconut", "fruit").
is_a("citrus fruit", "fruit").
is_a("tangerine", "citrus fruit").
is_a("onion", "vegetable").
is_a("pumpkin", "vegetable").
is_a("spinach", "vegetable").
is_a("broccoli", "vegetable").
is_a("mushroom", "vegetable").
is_a("carrot", "vegetable").
is_a("cabbage", "vegetable").
is_a("potato", "vegetable").
is_a("lettuce", "vegetable").
is_a("tomato", "vegetable").
is_a("beans", "vegetable").
is_a("squash", "vegetable").
is_a("cucumber", "vegetable").
is_a("eggplant", "vegetable").
is_a("celery", "vegetable").
is_a("pepper", "vegetable").
is_a("chili", "vegetable").
is_a("parsley", "vegetable").
is_a("sweet potato", "vegetable").
is_a("olive", "vegetable").
is_a("zucchini", "vegetable").
is_a("artichoke", "vegetable").
is_a("cauliflower", "vegetable").
is_a("avocado", "vegetable").
is_a("herb", "vegetable").
is_a("beet", "vegetable").
is_a("pea", "vegetable").
is_a("nut", "nut").
is_a("walnut", "nut").
is_a("pecan", "nut").
is_a("peanut", "nut").
is_a("pistachio", "nut").
is_a("almond", "nut").
is_a("dip", "condiment").
is_a("pesto", "condiment").
is_a("hummus", "condiment").
is_a("peanut butter", "condiment").
is_a("ginger", "condiment").
is_a("toast", "breakfast food").
is_a("cereal", "breakfast food").
is_a("doughnut", "breakfast food").
is_a("waffle", "breakfast food").
is_a("egg", "breakfast food").
is_a("pancake", "breakfast food").
is_a("beans", "side dishes").
is_a("broccoli", "side dishes").
is_a("potato", "side dishes").
is_a("salad", "side dishes").
is_a("cabbage", "side dishes").
is_a("squash", "side dishes").
is_a("mushroom", "side dishes").
is_a("fries", "side dishes").
is_a("maize", "staple food").
is_a("millet", "staple food").
is_a("sorghum", "staple food").
is_a("rice", "soft food").
is_a("ice cream", "soft food").
is_a("chocolate", "soft food").
is_a("cake", "soft food").
is_a("cupcake", "soft food").
is_a("cheesecake", "soft food").
is_a("pie", "soft food").
is_a("pudding", "soft food").
is_a("sauce", "soft food").
is_a("dip", "soft food").
is_a("sugar", "soft food").
is_a("caramel", "soft food").
is_a("ketchup", "soft food").
is_a("pesto", "soft food").
is_a("gravy", "soft food").
is_a("guacamole", "soft food").
is_a("hummus", "soft food").
is_a("peanut butter", "soft food").
is_a("butter", "soft food").
is_a("syrup", "soft food").
is_a("mustard", "soft food").
is_a("meat", "solid food").
is_a("side dishes", "solid food").
is_a("fruit", "solid food").
is_a("main course", "solid food").
is_a("vegetable", "solid food").
is_a("pizza", "food").
is_a("sandwich", "food").
is_a("hot dog", "food").
is_a("noodles", "food").
is_a("pasta", "food").
is_a("donut", "food").
is_a("cupcake", "food").
is_a("bread", "food").
is_a("rice", "food").
is_a("cereal", "food").
is_a("chips", "food").
is_a("bun", "food").
is_a("cake", "food").
is_a("doughnut", "food").
is_a("fries", "food").
is_a("burger", "food").
is_a("hamburger", "food").
is_a("porridge", "food").
is_a("pie", "food").
is_a("vegetable", "food").
is_a("nut", "food").
is_a("meat", "food").
is_a("fruit", "food").
is_a("grains", "food").
is_a("side dishes", "food").
is_a("dessert", "food").
is_a("main course", "food").
is_a("breakfast food", "food").
is_a("milk", "drinks").
is_a("juice", "drinks").
is_a("soda", "drinks").
is_a("cola", "drinks").
is_a("cappuccino", "drinks").
is_a("milkshake", "drinks").
is_a("lemonade", "drinks").
is_a("liquor", "drinks").
is_a("alcohol", "drinks").
is_a("beer", "drinks").
is_a("wine", "drinks").
is_a("champagne", "drinks").
is_a("coffee", "drinks").
is_a("tea", "drinks").
is_a("water", "drinks").
is_a("soft drinks", "drinks").
is_a("alcoholic beverage", "drinks").
is_a("milk", "beverage").
is_a("juice", "beverage").
is_a("soda", "beverage").
is_a("cola", "beverage").
is_a("cappuccino", "beverage").
is_a("milkshake", "beverage").
is_a("lemonade", "beverage").
is_a("liquor", "beverage").
is_a("alcohol", "beverage").
is_a("beer", "beverage").
is_a("wine", "beverage").
is_a("champagne", "beverage").
is_a("coffee", "beverage").
is_a("tea", "beverage").
is_a("water", "beverage").
is_a("shelf", "furniture").
is_a("bookshelf", "furniture").
is_a("bookcase", "furniture").
is_a("drawer", "furniture").
is_a("entertainment center", "furniture").
is_a("medicine cabinet", "furniture").
is_a("table", "furniture").
is_a("end table", "furniture").
is_a("dining table", "furniture").
is_a("picnic table", "furniture").
is_a("side table", "furniture").
is_a("coffee table", "furniture").
is_a("banquet table", "furniture").
is_a("desk", "furniture").
is_a("computer desk", "furniture").
is_a("tv stand", "furniture").
is_a("bed", "furniture").
is_a("mattress", "furniture").
is_a("nightstand", "furniture").
is_a("counter", "furniture").
is_a("blind", "furniture").
is_a("cabinet", "furniture").
is_a("wardrobe", "furniture").
is_a("chair", "furniture").
is_a("armchair", "furniture").
is_a("folding chair", "furniture").
is_a("beach chair", "furniture").
is_a("office chair", "furniture").
is_a("recliner", "furniture").
is_a("bench", "furniture").
is_a("stool", "furniture").
is_a("bar stool", "furniture").
is_a("seat", "furniture").
is_a("couch", "furniture").
is_a("sofa", "furniture").
is_a("ottoman", "furniture").
is_a("closet", "furniture").
is_a("dresser", "furniture").
is_a("cupboard", "furniture").
is_a("lamp", "furniture").
is_a("spatula", "kitchenware").
is_a("colander", "kitchenware").
is_a("tongs", "kitchenware").
is_a("blade", "kitchenware").
is_a("cutting board", "kitchenware").
is_a("foil", "kitchenware").
is_a("dishwasher", "kitchenware").
is_a("sink", "kitchenware").
is_a("microwave", "kitchenware").
is_a("blender", "kitchenware").
is_a("toaster", "kitchenware").
is_a("oven", "kitchenware").
is_a("stove", "kitchenware").
is_a("grill", "kitchenware").
is_a("fridge", "kitchenware").
is_a("container", "kitchenware").
is_a("pot", "kitchenware").
is_a("kettle", "kitchenware").
is_a("mixer", "kitchenware").
is_a("electrical appliance", "home appliance").
is_a("kitchenware", "home appliance").
is_a("radiator", "home appliance").
is_a("stove", "home appliance").
is_a("gas stove", "home appliance").
is_a("spoon", "tableware").
is_a("knife", "tableware").
is_a("fork", "tableware").
is_a("chopsticks", "tableware").
is_a("tray", "tableware").
is_a("pizza tray", "tableware").
is_a("placemat", "tableware").
is_a("dishes", "tableware").
is_a("napkin", "tableware").
is_a("plate", "tableware").
is_a("saucer", "tableware").
is_a("cup", "tableware").
is_a("coffee cup", "tableware").
is_a("glass", "tableware").
is_a("wine glass", "tableware").
is_a("water glass", "tableware").
is_a("mug", "tableware").
is_a("beer mug", "tableware").
is_a("coffee mug", "tableware").
is_a("bowl", "tableware").
is_a("straw", "tableware").
is_a("tablecloth", "tableware").
is_a("cloth", "tableware").
is_a("basket", "tableware").
is_a("candle", "tableware").
is_a("can", "tableware").
is_a("salt shaker", "tableware").
is_a("pepper shaker", "tableware").
is_a("vessel", "tableware").
is_a("vase", "tableware").
is_a("eating utensil", "utensil").
is_a("kitchen utensil", "utensil").
is_a("shampoo bottle", "bottle").
is_a("perfume bottle", "bottle").
is_a("soap bottle", "bottle").
is_a("ketchup bottle", "bottle").
is_a("spray bottle", "bottle").
is_a("mustard bottle", "bottle").
is_a("water bottle", "bottle").
is_a("wine bottle", "bottle").
is_a("soda bottle", "bottle").
is_a("beer bottle", "bottle").
is_a("adidas", "logo").
is_a("nike", "logo").
is_a("apple logo", "logo").
is_a("adidas", "brand").
is_a("nike", "brand").
is_a("laptop", "computer").
is_a("bedspread", "bedding").
is_a("pillow", "bedding").
is_a("duvet", "bedding").
is_a("duvet cover", "bedding").
is_a("quilt", "bedding").
is_a("sheet", "bedding").
is_a("blanket", "bedding").
is_a("mattress", "bedding").
is_a("teddy bear", "toy").
is_a("rubber duck", "toy").
is_a("lego", "toy").
is_a("stuffed bear", "toy").
is_a("stuffed dog", "toy").
is_a("stuffed animal", "toy").
is_a("toy car", "toy").
is_a("balloon", "toy").
is_a("doll", "toy").
is_a("kite", "toy").
is_a("frisbee", "toy").
is_a("teddy bear", "stuffed animal").
is_a("stuffed bear", "stuffed animal").
is_a("stuffed dog", "stuffed animal").
is_a("cable", "electronic device").
is_a("hard drive", "electronic device").
is_a("charger", "electronic device").
is_a("cd", "electronic device").
is_a("remote", "electronic device").
is_a("controller", "electronic device").
is_a("telephone", "electronic device").
is_a("bicycle", "vehicle").
is_a("cart", "vehicle").
is_a("wagon", "vehicle").
is_a("carriage", "vehicle").
is_a("stroller", "vehicle").
is_a("motorcycle", "vehicle").
is_a("scooter", "vehicle").
is_a("subway", "vehicle").
is_a("train", "vehicle").
is_a("car", "vehicle").
is_a("planter", "vehicle").
is_a("tractor", "vehicle").
is_a("crane", "vehicle").
is_a("aircraft", "vehicle").
is_a("watercraft", "vehicle").
is_a("trailer", "vehicle").
is_a("truck", "vehicle").
is_a("fire truck", "vehicle").
is_a("bus", "vehicle").
is_a("school bus", "vehicle").
is_a("ambulance", "vehicle").
is_a("double decker", "vehicle").
is_a("taxi", "vehicle").
is_a("sedan", "car").
is_a("minivan", "car").
is_a("van", "car").
is_a("pickup", "car").
is_a("jeep", "car").
is_a("suv", "car").
is_a("micro", "car").
is_a("hatchback", "car").
is_a("coupe", "car").
is_a("station wagon", "car").
is_a("roadster", "car").
is_a("cabriolet", "car").
is_a("muscle car", "car").
is_a("sport car", "car").
is_a("super car", "car").
is_a("limousine", "car").
is_a("cuv", "car").
is_a("campervan", "car").
is_a("engine", "part of vehicle").
is_a("cargo", "part of vehicle").
is_a("steering wheel", "part of vehicle").
is_a("kickstand", "part of vehicle").
is_a("wheel", "part of vehicle").
is_a("tire", "part of vehicle").
is_a("windshield", "part of vehicle").
is_a("propeller", "part of vehicle").
is_a("paddle", "part of vehicle").
is_a("locomotive", "part of vehicle").
is_a("letters", "symbol").
is_a("words", "symbol").
is_a("numbers", "symbol").
is_a("roman numerals", "symbol").
is_a("snowboard", "sports equipment").
is_a("skateboard", "sports equipment").
is_a("surfboard", "sports equipment").
is_a("skis", "sports equipment").
is_a("frisbee", "sports equipment").
is_a("ball", "sports equipment").
is_a("tennis ball", "sports equipment").
is_a("soccer ball", "sports equipment").
is_a("kite", "sports equipment").
is_a("hurdle", "sports equipment").
is_a("racket", "sports equipment").
is_a("baseball bat", "sports equipment").
is_a("baseball helmet", "sports equipment").
is_a("baseball glove", "sports equipment").
is_a("baseball mitt", "sports equipment").
is_a("goggles", "sports equipment").
is_a("bedroom", "place").
is_a("living room", "place").
is_a("dining room", "place").
is_a("kitchen", "place").
is_a("bathroom", "place").
is_a("alcove", "place").
is_a("attic", "place").
is_a("basement", "place").
is_a("closet", "place").
is_a("home office", "place").
is_a("pantry", "place").
is_a("shower", "place").
is_a("staircase", "place").
is_a("tent", "place").
is_a("hall", "place").
is_a("balcony", "place").
is_a("patio", "place").
is_a("factory", "place").
is_a("lab", "place").
is_a("office", "place").
is_a("classroom", "place").
is_a("building", "place").
is_a("apartment", "place").
is_a("restaurant", "place").
is_a("alley", "place").
is_a("bar", "place").
is_a("supermarket", "place").
is_a("shop", "place").
is_a("market", "place").
is_a("store", "place").
is_a("mall", "place").
is_a("plaza", "place").
is_a("theater", "place").
is_a("courtyard", "place").
is_a("gas station", "place").
is_a("restroom", "place").
is_a("library", "place").
is_a("dormitory", "place").
is_a("aquarium", "place").
is_a("school", "place").
is_a("bank", "place").
is_a("hospital", "place").
is_a("casino", "place").
is_a("baseball field", "place").
is_a("bleachers", "place").
is_a("dugout", "place").
is_a("football field", "place").
is_a("soccer field", "place").
is_a("golf course", "place").
is_a("stadium", "place").
is_a("court", "place").
is_a("tennis court", "place").
is_a("ski lift", "place").
is_a("ski slope", "place").
is_a("swimming pool", "place").
is_a("pool", "place").
is_a("playground", "place").
is_a("park", "place").
is_a("resort", "place").
is_a("skate park", "place").
is_a("station", "place").
is_a("bus station", "place").
is_a("train station", "place").
is_a("bus stop", "place").
is_a("airport", "place").
is_a("tarmac", "place").
is_a("freeway", "place").
is_a("street", "place").
is_a("tunnel", "place").
is_a("highway", "place").
is_a("driveway", "place").
is_a("road", "place").
is_a("crosswalk", "place").
is_a("overpass", "place").
is_a("runway", "place").
is_a("railway", "place").
is_a("parking lot", "place").
is_a("track", "place").
is_a("bridge", "place").
is_a("airfield", "place").
is_a("shore", "place").
is_a("beach", "place").
is_a("harbor", "place").
is_a("jetty", "place").
is_a("dock", "place").
is_a("pier", "place").
is_a("sidewalk", "place").
is_a("lane", "place").
is_a("curb", "place").
is_a("crosswalkhouse", "place").
is_a("home", "place").
is_a("hotel", "place").
is_a("farm", "place").
is_a("barn", "place").
is_a("garage", "place").
is_a("corn field", "place").
is_a("corral", "place").
is_a("garden", "place").
is_a("orchard", "place").
is_a("tower", "place").
is_a("windmill", "place").
is_a("church", "place").
is_a("temple", "place").
is_a("chapel", "place").
is_a("shrine", "place").
is_a("lighthouse", "place").
is_a("clock tower", "place").
is_a("arch", "place").
is_a("dam", "place").
is_a("zoo", "place").
is_a("ocean", "place").
is_a("lake", "place").
is_a("pond", "place").
is_a("river", "place").
is_a("raft", "place").
is_a("creekswamp", "place").
is_a("waterfall", "place").
is_a("wave", "place").
is_a("canyon", "place").
is_a("cliff", "place").
is_a("desert", "place").
is_a("mountain", "place").
is_a("hill", "place").
is_a("valley", "place").
is_a("plain", "place").
is_a("air", "place").
is_a("land", "place").
is_a("sky", "place").
is_a("bamboo forest", "place").
is_a("forest", "place").
is_a("jungle", "place").
is_a("yard", "place").
is_a("field", "place").
is_a("savanna", "place").
is_a("bayou", "place").
is_a("city", "place").
is_a("downtown", "place").
is_a("wild", "place").
is_a("bedroom", "room").
is_a("living room", "room").
is_a("dining room", "room").
is_a("kitchen", "room").
is_a("bathroom", "room").
is_a("alcove", "room").
is_a("attic", "room").
is_a("basement", "room").
is_a("closet", "room").
is_a("home office", "room").
is_a("office", "room").
is_a("pantry", "room").
is_a("shower", "room").
is_a("staircase", "room").
is_a("hall", "room").
is_a("balcony", "room").
is_a("hydrant", "object").
is_a("manhole cover", "object").
is_a("fountain", "object").
is_a("line", "object").
is_a("parking meter", "object").
is_a("mailbox", "object").
is_a("pole", "object").
is_a("street light", "object").
is_a("sign", "object").
is_a("street sign", "object").
is_a("stop sign", "object").
is_a("traffic sign", "object").
is_a("parking signtraffic light", "object").
is_a("bench", "object").
is_a("trash can", "object").
is_a("cone", "object").
is_a("dispenser", "object").
is_a("vending machine", "object").
is_a("toolbox", "object").
is_a("buoy", "object").
is_a("dumpster", "object").
is_a("garbage", "object").
is_a("umbrella", "object").
is_a("canopy", "object").
is_a("backpack", "object").
is_a("luggage", "object").
is_a("purse", "object").
is_a("wallet", "object").
is_a("bag", "object").
is_a("handbag", "object").
is_a("shopping bag", "object").
is_a("trash bag", "object").
is_a("pouch", "object").
is_a("suitcase", "object").
is_a("box", "object").
is_a("crate", "object").
is_a("sack", "object").
is_a("cardboard", "object").
is_a("light bulb", "object").
is_a("christmas light", "object").
is_a("ceiling light", "object").
is_a("clock", "object").
is_a("alarm clock", "object").
is_a("gift", "object").
is_a("wheelchair", "object").
is_a("beach umbrella", "object").
is_a("parachute", "object").
is_a("feeder", "object").
is_a("fire extinguisher", "object").
is_a("tissue box", "object").
is_a("paper dispenser", "object").
is_a("soap dispenser", "object").
is_a("napkin dispenser", "object").
is_a("towel dispenser", "object").
is_a("spray can", "object").
is_a("paint brush", "object").
is_a("cash register", "object").
is_a("candle holder", "object").
is_a("bell", "object").
is_a("lock", "object").
is_a("cigarette", "object").
is_a("curtain", "object").
is_a("carpet", "object").
is_a("rug", "object").
is_a("thermometer", "object").
is_a("fence", "object").
is_a("barrier", "object").
is_a("stick", "object").
is_a("rope", "object").
is_a("chain", "object").
is_a("hook", "object").
is_a("cage", "object").
is_a("chalk", "object").
is_a("chalkboard", "object").
is_a("money", "object").
is_a("coin", "object").
is_a("shield", "object").
is_a("armor", "object").
is_a("seat belt", "object").
is_a("chimney", "object").
is_a("fishing pole", "object").
is_a("bottle", "object").
is_a("bandage", "object").
is_a("lipstick", "object").
is_a("wig", "object").
is_a("shaving cream", "object").
is_a("deodorant", "object").
is_a("lotion", "object").
is_a("sink", "object").
is_a("faucet", "object").
is_a("fireplace", "object").
is_a("shower", "object").
is_a("fan", "object").
is_a("light switch", "object").
is_a("figure", "object").
is_a("frame", "object").
is_a("picture frame", "object").
is_a("door frame", "object").
is_a("window frame", "object").
is_a("lamp", "object").
is_a("table lamp", "object").
is_a("desk lamp", "object").
is_a("lamps", "object").
is_a("floor lamp", "object").
is_a("sconce", "object").
is_a("chandelier", "object").
is_a("bathtub", "object").
is_a("urinal", "object").
is_a("soap dish", "object").
is_a("fans", "object").
is_a("string", "object").
is_a("shade", "object").
is_a("tarp", "object").
is_a("handle", "object").
is_a("knob", "object").
is_a("hammer", "object").
is_a("screw", "object").
is_a("broom", "object").
is_a("sponge", "object").
is_a("cane", "object").
is_a("knife block", "object").
is_a("waste basket", "object").
is_a("satellite dish", "object").
is_a("shopping cart", "object").
is_a("tape", "object").
is_a("cord", "object").
is_a("power line", "object").
is_a("book", "object").
is_a("newspaper", "object").
is_a("magazine", "object").
is_a("paper", "object").
is_a("notebook", "object").
is_a("notepad", "object").
is_a("cookbook", "object").
is_a("map", "object").
is_a("envelope", "object").
is_a("pen", "object").
is_a("pencil", "object").
is_a("marker", "object").
is_a("crayon", "object").
is_a("pencil sharpener", "object").
is_a("ruler", "object").
is_a("binder", "object").
is_a("scissors", "object").
is_a("stapler", "object").
is_a("staples", "object").
is_a("glue stick", "object").
is_a("clip", "object").
is_a("folder", "object").
is_a("briefcase", "object").
is_a("vehicle", "object").
is_a("sports equipment", "object").
is_a("artwork", "object").
is_a("writing implement", "object").
is_a("electronic device", "object").
is_a("instruments", "object").
is_a("toy", "object").
is_a("bedding", "object").
is_a("weapon", "object").
is_a("utensil", "object").
is_a("tableware", "object").
is_a("home appliance", "object").
is_a("kitchenware", "object").
is_a("furniture", "object").
is_a("food", "object").
is_a("drinks", "object").
is_a("beverage", "object").
is_a("accessory", "object").
is_a("clothing", "object").
is_a("animal", "object").
is_a("plant", "object").
is_a("alpaca", "herbivore").
is_a("amphibian", "class").
is_a("odd-toed ungulate", "order").
is_a("even-toed ungulate", "order").
is_a("primate", "order").
is_a("carnivoran", "order").
is_a("canidae", "family").
is_a("perfume", "liquid").
is_a("cosmetic", "toiletry").
is_a("antiperspirant", "toiletry").
is_a("eyebrow pencil", "cosmetic").
is_a("face powder", "cosmetic").
is_a("facial moisturizer", "cosmetic").
is_a("rouge", "cosmetic").
'''

KG_REL= '''
oa_rel("is used for", "floor", "standing on").
oa_rel("is used for", "shelf", "storing foods").
oa_rel("is used for", "wall", "holding up roof").
oa_rel("is used for", "chair", "sitting on").
oa_rel("is used for", "table", "holding things").
oa_rel("is used for", "bookshelf", "storing magazines").
oa_rel("is used for", "bookshelf", "holding books").
oa_rel("can", "vase", "holds flowers").
oa_rel("usually appears in", "book", "office").
oa_rel("is used for", "fireplace", "burning things").
oa_rel("is used for", "window", "letting light in").
oa_rel("can", "grass", "turn brown").
oa_rel("can", "fish", "navigate via polarised light").
oa_rel("can", "cat", "eat fish").
oa_rel("is used for", "bridge", "crossing valley").
oa_rel("is used for", "stool", "reaching high places").
oa_rel("is", "screen", "electric").
oa_rel("is used for", "carpet", "saving floor").
oa_rel("is used for", "store", "buying and selling").
oa_rel("is used for", "box", "putting things in").
oa_rel("is used for", "dresser", "supporting mirror").
oa_rel("can", "tree", "shade car").
oa_rel("can be", "paper", "cut").
oa_rel("is used for", "table", "writing at").
oa_rel("is used for", "pot", "cooking stew").
oa_rel("can", "gas stove", "heat pot").
oa_rel("is used for", "patio", "sitting outside").
oa_rel("can", "umbrella", "protect you from sun").
oa_rel("can be", "door", "opened or closed").
oa_rel("is used for", "seat", "sitting on").
oa_rel("can", "remote", "control tv").
oa_rel("is used for", "remote", "remotely controlling TV").
oa_rel("is", "console", "electric").
oa_rel("can", "car", "travel on road").
oa_rel("is used for", "window", "letting fresh air in").
oa_rel("is used for", "car", "transporting handful of people").
oa_rel("is used for", "road", "driving car on").
oa_rel("can", "cart", "follow horse").
oa_rel("can", "horse", "pull cart").
oa_rel("is used for", "luggage", "carrying things").
oa_rel("usually appears in", "shelf", "bedroom").
oa_rel("has", "cupcake", "starch").
oa_rel("is used for", "container", "holding foods").
oa_rel("is", "chocolate", "sticky").
oa_rel("is used for", "tray", "holding food").
oa_rel("is used for", "paper", "drawing on").
oa_rel("is made from", "hot dog", "flour").
oa_rel("usually appears in", "blanket", "bedroom").
oa_rel("is made from", "cinnamon roll", "flour").
oa_rel("is used for", "sugar", "sweetening food").
oa_rel("is used for", "basket", "carrying something").
oa_rel("usually appears in", "tray", "restaurant").
oa_rel("is made from", "dough", "flour").
oa_rel("requires", "sauteing", "pan").
oa_rel("is used for", "table", "eating at").
oa_rel("is used for", "window", "looking outside").
oa_rel("is used for", "bowl", "holding fruit").
oa_rel("is used for", "wall", "hanging picture").
oa_rel("is used for", "glass", "holding drinks").
oa_rel("can", "hammer", "break glass").
oa_rel("can", "knife", "cut you").
oa_rel("requires", "slicing", "knife").
oa_rel("has", "cake", "starch").
oa_rel("usually appears in", "tub", "bathroom").
oa_rel("is used for", "plate", "holding pizza").
oa_rel("is used for", "garage", "parking car").
oa_rel("is used for", "bar", "meeting friends").
oa_rel("can", "suv", "travel on road").
oa_rel("is used for", "bar", "getting drunk").
oa_rel("is", "alcohol", "harmful").
oa_rel("can", "bottle", "hold liquid").
oa_rel("is used for", "drinks", "satisfying thirst").
oa_rel("is used for", "stool", "tying shoes").
oa_rel("can", "vacuum", "clean floor").
oa_rel("can", "wine", "age in bottle").
oa_rel("is", "wine", "liquid").
oa_rel("is used for", "lamp", "lighting room").
oa_rel("usually appears in", "wine glass", "restaurant").
oa_rel("is used for", "tablecloth", "keeping table clean").
oa_rel("is used for", "tablecloth", "decoration").
oa_rel("usually appears in", "plate", "restaurant").
oa_rel("is used for", "barn", "keeping animals").
oa_rel("is used for", "bedroom", "sleeping").
oa_rel("usually appears in", "bed", "bedroom").
oa_rel("is used for", "pillow", "make seat softer").
oa_rel("usually appears in", "mirror", "bathroom").
oa_rel("can", "curtain", "keep light out of room").
oa_rel("can", "tv", "display images").
oa_rel("is used for", "dresser", "storing cloth").
oa_rel("is used for", "door", "making room private").
oa_rel("is used for", "wall", "divide open space into smaller areas").
oa_rel("is", "lamp", "electric").
oa_rel("is used for", "tv", "entertainment").
oa_rel("can", "truck", "pull cars").
oa_rel("can", "cart", "transport things").
oa_rel("is used for", "boat", "transporting people").
oa_rel("can", "tree", "shade lawn").
oa_rel("is", "water", "fluid").
oa_rel("is used for", "bicycle", "transporting people").
oa_rel("is used for", "street", "transportation").
oa_rel("is used for", "bottle", "holding juice").
oa_rel("is used for", "table", "playing game at").
oa_rel("is used for", "fireplace", "heating home").
oa_rel("is used for", "car", "transporting people").
oa_rel("is used for", "driveway", "transportation").
oa_rel("is", "computer", "electric").
oa_rel("is used for", "table", "putting things on").
oa_rel("is used for", "clock", "measuring passage of time").
oa_rel("is used for", "necklace", "decoration").
oa_rel("is used for", "book", "learning").
oa_rel("is used for", "fence", "keeping pets in").
oa_rel("is used for", "field", "grazing animals").
oa_rel("is used for", "floor", "walking on").
oa_rel("can", "grass", "grow on hill").
oa_rel("can", "car", "transport people").
oa_rel("usually appears in", "coffee cup", "dining room").
oa_rel("is used for", "couch", "lying on").
oa_rel("usually appears in", "switch", "bedroom").
oa_rel("is used for", "rug", "covering area near front door").
oa_rel("is used for", "rug", "covering floor").
oa_rel("requires", "sauteing", "skillet").
oa_rel("is used for", "stove", "boiling water").
oa_rel("is used for", "fence", "enclosing space").
oa_rel("can", "bear", "climb tree").
oa_rel("is used for", "book", "reading for pleasure").
oa_rel("is used for", "bar", "meeting people").
oa_rel("is used for", "road", "transportation").
oa_rel("can", "bus", "carry passengers").
oa_rel("can", "bus", "travel on road").
oa_rel("usually appears in", "mirror", "bedroom").
oa_rel("is used for", "van", "transporting goods").
oa_rel("is used for", "phone", "communicating").
oa_rel("usually appears in", "paper", "office").
oa_rel("can be", "food", "eaten").
oa_rel("is used for", "spoon", "eating food that isn't very solid").
oa_rel("is used for", "spoon", "scooping food").
oa_rel("is", "dessert", "sweet").
oa_rel("usually appears in", "steak", "dinner").
oa_rel("is", "water", "liquid").
oa_rel("can", "bowl", "keep water in").
oa_rel("requires", "cooking", "cooking utensils").
oa_rel("has", "watermelon", "vitamin C").
oa_rel("can be", "fruit", "eaten").
oa_rel("is used for", "wall", "hanging art work").
oa_rel("is", "ocean", "fluid").
oa_rel("can", "chicken", "be pet").
oa_rel("is used for", "bowl", "holding cereal").
oa_rel("usually appears in", "carrot", "salad").
oa_rel("usually appears in", "lettuce", "salad").
oa_rel("is", "sauce", "sticky").
oa_rel("is used for", "plate", "holding food").
oa_rel("is used for", "food", "eating").
oa_rel("has", "cabbage", "vitamin C").
oa_rel("is used for", "bar", "drinking alcohol").
oa_rel("has", "cola", "water").
oa_rel("usually appears in", "laptop", "office").
oa_rel("is used for", "desk", "putting computer on").
oa_rel("is used for", "bed", "lying down").
oa_rel("is used for", "mouse", "interfacing with computer").
oa_rel("is used for", "refrigerator", "chilling drinks").
oa_rel("is used for", "drinks", "drinking").
oa_rel("is used for", "street sign", "giving instructions to road users").
oa_rel("can", "tree", "grow new branches").
oa_rel("requires", "cooling off", "air conditioner").
oa_rel("can", "water", "feel wet").
oa_rel("is used for", "umbrella", "protection from rain").
oa_rel("is", "apple", "healthy").
oa_rel("can", "car", "move quickly").
oa_rel("can", "glass", "hold liquid").
oa_rel("can", "horse", "pull wagon").
oa_rel("usually appears in", "candle", "dining room").
oa_rel("is used for", "bench", "lying down").
oa_rel("can", "cup", "store liquid").
oa_rel("is used for", "curtain", "covering window").
oa_rel("can", "umbrella", "shield one from rain or sun").
oa_rel("is used for", "curtain", "blocking light").
oa_rel("is used for", "bowl", "holding beans").
oa_rel("can", "cat", "kill birds").
oa_rel("is used for", "sidewalk", "skating on").
oa_rel("is used for", "sidewalk", "walking dog").
oa_rel("usually appears in", "clip", "office").
oa_rel("usually appears in", "lid", "bathroom").
oa_rel("is", "zebra", "herbivorous").
oa_rel("is used for", "paper", "writing on").
oa_rel("is used for", "pen", "signing checks").
oa_rel("usually appears in", "monitor", "office").
oa_rel("is used for", "glove", "protecting hand").
oa_rel("can", "trailer", "travel on road").
oa_rel("can", "truck", "pull trailer").
oa_rel("has", "shrimp", "iron").
oa_rel("has", "shrimp", "vitamin D").
oa_rel("is", "lime", "sour").
oa_rel("has", "meat", "vitamin B").
oa_rel("is made from", "bacon", "pork").
oa_rel("requires", "making pizza", "sauce").
oa_rel("is used for", "bus station", "waiting for bus").
oa_rel("is used for", "van", "transporting handful of people").
oa_rel("can", "car", "spend gas").
oa_rel("is used for", "track", "subway to run on").
oa_rel("is used for", "train", "transporting goods").
oa_rel("is used for", "blinds", "keep out light from houses").
oa_rel("can", "computer", "help people").
oa_rel("usually appears in", "holder", "bathroom").
oa_rel("usually appears in", "mug", "restaurant").
oa_rel("is used for", "glasses", "improving eyesight").
oa_rel("is used for", "horse", "riding").
oa_rel("can", "tree", "grow branch").
oa_rel("has", "donut", "starch").
oa_rel("is", "frosting", "sweet").
oa_rel("is used for", "shoes", "protecting feet").
oa_rel("is used for", "handbag", "carrying things").
oa_rel("can", "cat", "see well in dark").
oa_rel("can", "bat", "hit baseball").
oa_rel("is used for", "tennis ball", "hitting with racket").
oa_rel("is used for", "sink", "washing up face").
oa_rel("can", "towel", "dry hair").
oa_rel("usually appears in", "soap", "bathroom").
oa_rel("is", "cake", "sweet").
oa_rel("is made from", "cheese", "milk").
oa_rel("is used for", "bowl", "holding food").
oa_rel("is", "dog", "soft").
oa_rel("is used for", "seat", "resting").
oa_rel("is used for", "hotel", "sleeping away from home").
oa_rel("is used for", "track", "trains to run on").
oa_rel("is used for", "motorcycle", "transporting people").
oa_rel("can", "cat", "clean itself often").
oa_rel("is used for", "ring", "decoration").
oa_rel("is used for", "umbrella", "keeping sun off you").
oa_rel("is used for", "boat", "floating and moving on water").
oa_rel("has", "bun", "starch").
oa_rel("is used for", "rug", "making floor warmer").
oa_rel("is used for", "rug", "covering just outside shower").
oa_rel("is used for", "cabinet", "storing glasses").
oa_rel("usually appears in", "fridge", "kitchen").
oa_rel("is used for", "kitchen", "cooking food").
oa_rel("can", "butterfly", "fly").
oa_rel("is used for", "blanket", "sleeping under").
oa_rel("is used for", "pillow", "sleeping").
oa_rel("is used for", "bench", "sitting on").
oa_rel("can", "elephant", "lift logs from ground").
oa_rel("has", "elephant", "trunk").
oa_rel("can", "grass", "stain pants").
oa_rel("can", "grass", "continue to grow").
oa_rel("can", "elephant", "carry trunk").
oa_rel("can", "bear", "fish with it's paw").
oa_rel("is used for", "fence", "containing animals").
oa_rel("can", "snow", "be packed into ball").
oa_rel("can", "horse", "rest standing up").
oa_rel("can", "jeep", "climb hills").
oa_rel("is used for", "watch", "measuring passage of time").
oa_rel("is used for", "computer", "studying").
oa_rel("is", "laptop", "electric").
oa_rel("is used for", "candle", "decoration").
oa_rel("is used for", "mouse", "controlling computer").
oa_rel("has", "giraffe", "marsupium").
oa_rel("is used for", "hot dog", "eating").
oa_rel("has", "beer", "alcohol").
oa_rel("is used for", "napkin", "wiping mouth").
oa_rel("is used for", "carpet", "decorating apartment").
oa_rel("is used for", "keyboard", "interfacing with computer").
oa_rel("is used for", "monitor", "displaying images").
oa_rel("is used for", "stool", "sitting on").
oa_rel("is used for", "handbag", "storing things").
oa_rel("is used for", "lane", "driving car on").
oa_rel("is used for", "desk", "reading at").
oa_rel("is used for", "mat", "protection something").
oa_rel("is used for", "bus", "transporting people").
oa_rel("is used for", "bus", "mass transit for city").
oa_rel("is used for", "trash can", "storing trash").
oa_rel("can", "turtle", "live much longer than people").
oa_rel("is", "candy", "sweet").
oa_rel("usually appears in", "chair", "bedroom").
oa_rel("is used for", "carpet", "walking on").
oa_rel("usually appears in", "chalkboard", "classroom").
oa_rel("is used for", "scooter", "transporting handful of people").
oa_rel("is used for", "toy", "having fun").
oa_rel("is used for", "fan", "circulating air").
oa_rel("is", "charger", "electric").
oa_rel("is used for", "chair", "resting").
oa_rel("is used for", "bookshelf", "storing books").
oa_rel("can", "bottle", "store wine").
oa_rel("can", "monitor", "display images").
oa_rel("is used for", "keyboard", "controlling computer").
oa_rel("is used for", "keyboard", "entering text").
oa_rel("can", "radiator", "heat room").
oa_rel("can", "fan", "cool air").
oa_rel("usually appears in", "desk", "bedroom").
oa_rel("is", "projector", "electric").
oa_rel("is used for", "computer", "data storage").
oa_rel("is used for", "desk", "placing something on").
oa_rel("is used for", "keyboard", "entering data").
oa_rel("is", "headphones", "electric").
oa_rel("is used for", "window", "keeping cold air out").
oa_rel("can", "computer", "save files on disk").
oa_rel("is", "monitor", "electric").
oa_rel("is used for", "office", "business").
oa_rel("is used for", "door", "entering or exiting area").
oa_rel("is used for", "classroom", "learning").
oa_rel("can", "monitors", "show text").
oa_rel("is used for", "theater", "watching movie").
oa_rel("is used for", "balcony", "viewing, resting, or eating at").
oa_rel("is used for", "spatula", "turning food").
oa_rel("is used for", "candle", "creating ambience").
oa_rel("can", "car", "carry few persons").
oa_rel("can", "rat", "eat wires").
oa_rel("usually appears in", "office chair", "office").
oa_rel("can", "cup", "hold liquids").
oa_rel("is used for", "restaurant", "drinking").
oa_rel("is used for", "restaurant", "meeting people").
oa_rel("is made from", "paper", "wood").
oa_rel("is used for", "shirt", "covering upperbody").
oa_rel("can", "printer", "print pictures").
oa_rel("can", "minivan", "travel on road").
oa_rel("is used for", "fountain", "decoration").
oa_rel("can", "poodle", "live in house").
oa_rel("usually appears in", "pot", "kitchen").
oa_rel("is used for", "kitchen", "storing food").
oa_rel("is used for", "sheet", "covering bed").
oa_rel("is used for", "hospital", "delivering babies").
oa_rel("is used for", "bed", "sitting on").
oa_rel("is used for", "cabinet", "storing dishes").
oa_rel("can", "bus", "transport people").
oa_rel("can", "bottle", "hold water").
oa_rel("can", "horse", "be tamed").
oa_rel("can", "airplane", "arrive at airport").
oa_rel("is used for", "truck", "carrying cargo").
oa_rel("is used for", "airport", "waiting for airplane").
oa_rel("is used for", "stop sign", "controlling traffic").
oa_rel("is used for", "boat", "transportation at sea").
oa_rel("can", "dog", "guard house").
oa_rel("requires", "washing dishes", "faucet").
oa_rel("is used for", "straw", "drinking beverage").
oa_rel("is made from", "donut", "flour").
oa_rel("is made from", "pastry", "flour").
oa_rel("usually appears in", "glass", "restaurant").
oa_rel("is used for", "phone", "sending email").
oa_rel("can", "truck", "travel on road").
oa_rel("can", "water", "act as reflector").
oa_rel("is used for", "raft", "keeping people out of water").
oa_rel("is used for", "raft", "traveling on water").
oa_rel("can", "dog", "detect odors better than humans can").
oa_rel("can", "bird", "live in house").
oa_rel("usually appears in", "customer", "bar").
oa_rel("is", "banana", "sweet").
oa_rel("is used for", "sofa", "lying down").
oa_rel("is", "oven", "electric").
oa_rel("requires", "making pizza", "oven").
oa_rel("is used for", "lamp", "illuminating area").
oa_rel("is used for", "vase", "holding flowers").
oa_rel("can be", "cake", "cut").
oa_rel("can", "bus", "run").
oa_rel("is used for", "bus stop", "waiting for bus").
oa_rel("is used for", "street light", "illuminating area").
oa_rel("can", "traffic light", "stop cars").
oa_rel("can", "horse", "carry people").
oa_rel("is used for", "motorcycle", "riding").
oa_rel("can", "ship", "carry cargo").
oa_rel("is used for", "ship", "keeping people out of water").
oa_rel("is used for", "boat", "traveling on water").
oa_rel("is made from", "chips", "potato").
oa_rel("requires", "measuring up", "measuring cup").
oa_rel("usually appears in", "dish soap", "bathroom").
oa_rel("is a sub-event of", "cleaning clothing", "operating washing machine").
oa_rel("can", "horse", "pull buggy to picnic").
oa_rel("can", "cart", "travel on road").
oa_rel("is used for", "street", "driving car on").
oa_rel("can", "bus", "drive down street").
oa_rel("is used for", "shelf", "holding books").
oa_rel("is used for", "restaurant", "selling food").
oa_rel("can", "refrigerator", "stock food").
oa_rel("is used for", "refrigerator", "freezing food").
oa_rel("is used for", "couch", "sitting on").
oa_rel("usually appears in", "sofa", "living room").
oa_rel("can", "dog", "live in house").
oa_rel("usually appears in", "wine glass", "bar").
oa_rel("has", "corn", "vitamin B").
oa_rel("can", "airplane", "go fast").
oa_rel("is used for", "wine", "getting drunk").
oa_rel("is used for", "sink", "washing hands").
oa_rel("is used for", "horse", "transporting people").
oa_rel("can", "horse", "jump higher than people").
oa_rel("usually appears in", "plate", "kitchen").
oa_rel("usually appears in", "lamp", "bedroom").
oa_rel("is used for", "bed", "napping on").
oa_rel("is used for", "sheet", "covering mattress").
oa_rel("is used for", "fan", "cooling air").
oa_rel("can", "refrigerator", "keep food cold").
oa_rel("is used for", "blanket", "keeping warm at night").
oa_rel("is used for", "door", "controlling access").
oa_rel("usually appears in", "notebook", "office").
oa_rel("is used for", "kitchen", "preparing food").
oa_rel("requires", "cleaning house", "pail").
oa_rel("can", "pitcher", "throw fast ball").
oa_rel("is", "radiator", "electric").
oa_rel("usually appears in", "christmas tree", "christmas").
oa_rel("has", "wine", "alcohol").
oa_rel("is used for", "sidewalk", "riding skateboards").
oa_rel("is used for", "bottle", "holding drinks").
oa_rel("is used for", "luggage", "storing clothes for trip").
oa_rel("is used for", "sunglasses", "protecting eyes from sunlight").
oa_rel("is used for", "sidewalk", "walking on").
oa_rel("is used for", "clock", "knowing time").
oa_rel("is used for", "bridge", "crossing river").
oa_rel("is used for", "river", "canoeing").
oa_rel("is used for", "curtain", "decorating room").
oa_rel("is used for", "bathroom", "brushing teeth").
oa_rel("can", "fire truck", "travel on road").
oa_rel("is used for", "market", "buying and selling").
oa_rel("is used for", "shelf", "storing items").
oa_rel("can", "boat", "travel over water").
oa_rel("is", "printer", "electric").
oa_rel("is used for", "teddy bear", "cuddling").
oa_rel("can", "umbrella", "fold up").
oa_rel("is used for", "train", "transporting people").
oa_rel("can", "jeep", "travel on road").
oa_rel("is used for", "toy", "entertainment").
oa_rel("can", "batter", "strike out").
oa_rel("is used for", "helmet", "protecting head").
oa_rel("can", "horse", "be pet").
oa_rel("is used for", "motorcycle", "transporting handful of people").
oa_rel("is used for", "baseball bat", "hitting something").
oa_rel("is used for", "house", "dwelling").
oa_rel("can", "train", "arrive at station").
oa_rel("can", "bus", "transport many people at once").
oa_rel("is used for", "crane", "transporting people").
oa_rel("can", "taxi", "travel on road").
oa_rel("is used for", "cabinet", "storing pills").
oa_rel("usually appears in", "bucket", "bathroom").
oa_rel("is", "xbox controller", "electric").
oa_rel("is used for", "goggles", "preventing particulates from striking eyes").
oa_rel("can", "helmet", "protect head from impact").
oa_rel("is used for", "watch", "knowing time").
oa_rel("usually appears in", "calculator", "office").
oa_rel("can be", "window", "opened or closed").
oa_rel("is", "light bulb", "electric").
oa_rel("is used for", "snow", "skiing on").
oa_rel("usually appears in", "napkin", "dining room").
oa_rel("usually appears in", "dishes", "restaurant").
oa_rel("can", "dog", "guard building").
oa_rel("is used for", "steering wheel", "controlling direction car turns").
oa_rel("can", "van", "travel on road").
oa_rel("can", "airplane", "transport many people at once").
oa_rel("usually appears in", "fireplace", "living room").
oa_rel("is", "wii controller", "electric").
oa_rel("is used for", "guitar", "playing music").
oa_rel("is", "speaker", "electric").
oa_rel("is used for", "boat", "fishing").
oa_rel("can", "steering wheel", "control direction of car").
oa_rel("is used for", "car", "transportation").
oa_rel("is used for", "fork", "moving food to mouth").
oa_rel("usually appears in", "water glass", "bar").
oa_rel("is used for", "bench", "resting").
oa_rel("can", "boat", "swim on water").
oa_rel("usually appears in", "menu", "restaurant").
oa_rel("can", "bull", "charge matador").
oa_rel("is used for", "speaker", "listening to music").
oa_rel("can", "hammers", "nail wood").
oa_rel("can", "monkeys", "use tool").
oa_rel("is used for", "pot", "holding water").
oa_rel("is", "elephant", "herbivorous").
oa_rel("is used for", "vehicle", "transportation").
oa_rel("is used for", "bicycle", "riding").
oa_rel("can", "truck", "carry cargo").
oa_rel("can", "stethoscopes", "listen to heart").
oa_rel("is used for", "suitcase", "packing clothes for trip").
oa_rel("is", "fruit", "healthy").
oa_rel("has", "grapefruit", "vitamin C").
oa_rel("can", "airplane", "cross ocean").
oa_rel("is used for", "runway", "aircraft takeoff").
oa_rel("is used for", "shower", "cleaning body").
oa_rel("is used for", "bathroom", "taking bath").
oa_rel("is used for", "sink", "cleaning dishes").
oa_rel("is used for", "menu", "ordering food").
oa_rel("can", "kite", "fly").
oa_rel("can", "truck", "move heavy loads").
oa_rel("is used for", "blanket", "covering things").
oa_rel("is used for", "church", "weddings").
oa_rel("is used for", "court", "playing tennis").
oa_rel("is used for", "bus", "transporting lots of people").
oa_rel("can", "lamp", "illuminate").
oa_rel("is used for", "computer", "finding information").
oa_rel("usually appears in", "salt shaker", "dining room").
oa_rel("is used for", "rug", "standing on").
oa_rel("is used for", "camera", "taking pictures").
oa_rel("usually appears in", "candle", "restaurant").
oa_rel("is used for", "door", "separating rooms").
oa_rel("can", "knife", "cut that cake").
oa_rel("usually appears in", "spoon", "dining room").
oa_rel("is used for", "bowl", "holding apples").
oa_rel("can", "knives", "cut food").
oa_rel("can", "oven", "roast").
oa_rel("is a sub-event of", "baking cake", "preheating oven").
oa_rel("is used for", "pot", "holding liquid").
oa_rel("is used for", "soap", "cleaning somethings").
oa_rel("requires", "cleaning clothing", "soap").
oa_rel("has", "avocado", "vitamin B").
oa_rel("can be", "vegetable", "eaten").
oa_rel("is", "mouse", "electric").
oa_rel("is used for", "desk", "writing upon").
oa_rel("is used for", "keyboard", "typing").
oa_rel("usually appears in", "toothbrush", "bathroom").
oa_rel("usually appears in", "bathtub", "bathroom").
oa_rel("is used for", "luggage", "carrying clothes on trip").
oa_rel("is used for", "microwave", "cooking food fast").
oa_rel("can", "oven", "warm meal").
oa_rel("requires", "roasting", "oven").
oa_rel("can be", "shirt", "hung").
oa_rel("is used for", "home", "dwelling").
oa_rel("is used for", "fence", "marking property lines").
oa_rel("can", "baseball", "travel very fast").
oa_rel("is used for", "platform", "standing on").
oa_rel("is", "wine", "fluid").
oa_rel("is used for", "glasses", "correcting vision").
oa_rel("can", "bench", "seat people").
oa_rel("usually appears in", "toilet brush", "bathroom").
oa_rel("is used for", "bathroom", "peeing").
oa_rel("requires", "washing dishes", "sink").
oa_rel("is used for", "refrigerator", "keeping food cold").
oa_rel("is", "stove", "electric").
oa_rel("usually appears in", "veil", "wedding").
oa_rel("is", "cigarette", "harmful").
oa_rel("is used for", "beverage", "drinking").
oa_rel("usually appears in", "soup", "dinner").
oa_rel("is used for", "bowl", "holding soup").
oa_rel("is used for", "airplane", "traversing skies").
oa_rel("is used for", "cup", "holding drinks").
oa_rel("has", "bread", "starch").
oa_rel("is used for", "phone", "surfing internet").
oa_rel("can", "truck", "ship goods").
oa_rel("is used for", "gas station", "buying gas").
oa_rel("can", "sedan", "travel on road").
oa_rel("is used for", "baseball field", "playing baseball").
oa_rel("is used for", "hospital", "healing sick people").
oa_rel("can", "heater", "heat room").
oa_rel("usually appears in", "drink", "bar").
oa_rel("usually appears in", "folder", "office").
oa_rel("is", "ocean", "liquid").
oa_rel("is used for", "meat", "eating").
oa_rel("is used for", "toilet", "depositing human waste").
oa_rel("is used for", "blanket", "covering bed").
oa_rel("can", "water", "dribble").
oa_rel("is used for", "bicycle", "transporting handful of people").
oa_rel("is used for", "belt", "holding pants").
oa_rel("can", "donkey", "carry load of supplies").
oa_rel("usually appears in", "closet", "bedroom").
oa_rel("is used for", "couch", "sleeping").
oa_rel("is", "radio", "electric").
oa_rel("can", "computer", "cost lot of money").
oa_rel("is used for", "bed", "resting").
oa_rel("can be", "umbrella", "opened or closed").
oa_rel("is used for", "dumpster", "storing trash").
oa_rel("is used for", "frisbee", "exercise").
oa_rel("can", "boat", "sail on pond").
oa_rel("requires", "baking bread", "oven").
oa_rel("is", "ice cream", "sticky").
oa_rel("can", "cup", "hold coffee").
oa_rel("has", "coffee", "water").
oa_rel("can", "dog", "shake hands").
oa_rel("is used for", "mall", "gathering shops").
oa_rel("is used for", "canoe", "traveling on water").
oa_rel("is used for", "canoe", "transporting people").
oa_rel("is used for", "ship", "transporting people").
oa_rel("usually appears in", "cup", "restaurant").
oa_rel("usually appears in", "mug", "dining room").
oa_rel("is", "bear", "carnivorous").
oa_rel("is used for", "couch", "taking nap").
oa_rel("is used for", "beverage", "satisfying thirst").
oa_rel("is used for", "airplane", "transporting people").
oa_rel("can", "airplane", "seat passengers").
oa_rel("usually appears in", "chair", "restaurant").
oa_rel("has", "broccoli", "vitamin B").
oa_rel("is", "ketchup", "fluid").
oa_rel("usually appears in", "elephant", "grassland").
oa_rel("is used for", "living room", "entertaining guests").
oa_rel("is used for", "sofa", "resting").
oa_rel("is used for", "ship", "transporting goods").
oa_rel("can", "ruler", "measure distance").
oa_rel("can", "docks", "shore boats").
oa_rel("is used for", "calculator", "doing mathematical calculations").
oa_rel("is made from", "ice cream", "milk").
oa_rel("is used for", "shelf", "storing dishes").
oa_rel("has", "juice", "water").
oa_rel("has", "milk", "vitamin B").
oa_rel("is", "ketchup", "sticky").
oa_rel("can", "hammer", "hit nail").
oa_rel("can", "axe", "chop wood").
oa_rel("is used for", "scissors", "cutting ribbons").
oa_rel("usually appears in", "grapes", "salad").
oa_rel("is used for", "traffic light", "controlling flows of traffic").
oa_rel("can", "washing machine", "clean clothes").
oa_rel("is used for", "desk", "working").
oa_rel("requires", "barbecue", "grill").
oa_rel("is used for", "blender", "mixing food").
oa_rel("is used for", "highway", "driving car on").
oa_rel("is used for", "frisbee", "catching").
oa_rel("can", "rain", "cause floods").
oa_rel("usually appears in", "candle", "birthday party").
oa_rel("is used for", "sailboat", "floating and moving on water").
oa_rel("is used for", "pot", "boiling water").
oa_rel("is used for", "pot", "planting plant").
oa_rel("is used for", "pen", "writing").
oa_rel("is used for", "fruit", "eating").
oa_rel("can", "steering wheel", "control car").
oa_rel("is used for", "airplane", "transporting goods").
oa_rel("is used for", "bathroom", "washing hands").
oa_rel("usually appears in", "tray", "dining room").
oa_rel("can", "pitcher", "strike out baseball hitter").
oa_rel("is used for", "earring", "decoration").
oa_rel("is used for", "bathtub", "cleaning body").
oa_rel("is used for", "truck", "transporting goods").
oa_rel("is used for", "doll", "playing with child").
oa_rel("is used for", "binder", "binding papers together").
oa_rel("usually appears in", "cup", "dining room").
oa_rel("is used for", "engine", "powering").
oa_rel("is used for", "home", "housing family").
oa_rel("is used for", "book", "getting knowledge").
oa_rel("can", "cat", "live in house").
oa_rel("usually appears in", "tissue box", "bathroom").
oa_rel("is used for", "bowl", "holding cream").
oa_rel("is", "tea", "liquid").
oa_rel("is used for", "bottle", "storing liquids").
oa_rel("can", "bird", "sing").
oa_rel("is used for", "fence", "creating privacy").
oa_rel("is", "apple", "sweet").
oa_rel("can", "knife", "cut apple").
oa_rel("is used for", "seat", "waiting").
oa_rel("is used for", "fruit", "making juice").
oa_rel("can", "fish", "swim").
oa_rel("can", "microwave", "heat food").
oa_rel("is used for", "cup", "handling liquid").
oa_rel("is used for", "blanket", "keeping warm when sleeping").
oa_rel("is used for", "carpet", "covering floor").
oa_rel("is used for", "market", "selling foodstuffs").
oa_rel("is used for", "wii", "playing video games").
oa_rel("can", "knife", "hurt you").
oa_rel("requires", "dicing", "knife").
oa_rel("can", "corn", "provide complex carbohydrates").
oa_rel("can", "truck", "use diesel fuel").
oa_rel("is used for", "stove", "heating food").
oa_rel("can", "boat", "take you to island").
oa_rel("can", "train", "carry freight").
oa_rel("is used for", "cake", "celebrating someones special event").
oa_rel("is used for", "bell", "getting people's attention").
oa_rel("can", "noodle", "provide complex carbohydrates").
oa_rel("is used for", "train", "transporting lots of people").
oa_rel("is used for", "refrigerator", "chilling food").
oa_rel("requires", "washing dishes", "dishwasher").
oa_rel("is used for", "menu", "listing choices").
oa_rel("is used for", "phone", "talking to someone").
oa_rel("is used for", "doll", "having fun").
oa_rel("is used for", "shelf", "storing books").
oa_rel("can", "toilet", "flush").
oa_rel("can", "stove", "heat pot").
oa_rel("usually appears in", "bowl", "kitchen").
oa_rel("is used for", "kitchen", "eating food").
oa_rel("usually appears in", "sink", "kitchen").
oa_rel("is used for", "kettle", "boiling water").
oa_rel("is used for", "sink", "soaking dishes").
oa_rel("is used for", "soap", "washing dishes").
oa_rel("usually appears in", "rag", "living room").
oa_rel("is used for", "sink", "washing dishes").
oa_rel("usually appears in", "blender", "kitchen").
oa_rel("requires", "blending", "blender").
oa_rel("usually appears in", "plate", "dining room").
oa_rel("requires", "toasting", "toaster").
oa_rel("can", "refrigerator", "store food for long times").
oa_rel("usually appears in", "oven", "kitchen").
oa_rel("requires", "baking", "oven").
oa_rel("usually appears in", "dishwasher", "kitchen").
oa_rel("is used for", "stove", "cooking stew").
oa_rel("requires", "making coffee", "coffee maker").
oa_rel("usually appears in", "towel rack", "bathroom").
oa_rel("is used for", "stove", "grilling steak").
oa_rel("can", "oven", "warm pie").
oa_rel("requires", "baking cake", "oven").
oa_rel("can", "stove", "heat pot of water").
oa_rel("requires", "boiling", "cooking pot").
oa_rel("usually appears in", "stove", "kitchen").
oa_rel("usually appears in", "placemat", "dining room").
oa_rel("usually appears in", "microwave", "kitchen").
oa_rel("is used for", "mug", "holding drinks").
oa_rel("usually appears in", "pan", "kitchen").
oa_rel("usually appears in", "towel", "bathroom").
oa_rel("is", "dryer", "electric").
oa_rel("is used for", "living room", "having party").
oa_rel("is a sub-event of", "cleaning clothing", "putting clothing in washer").
oa_rel("can", "coffee maker", "making coffee").
oa_rel("can", "microwave", "warm up coffee").
oa_rel("is used for", "camera", "photography").
oa_rel("requires", "mincing", "knife").
oa_rel("can", "bridge", "cross river").
oa_rel("usually appears in", "brush", "bathroom").
oa_rel("is used for", "river", "transportation").
oa_rel("can", "bird", "land on branch").
oa_rel("usually appears in", "coffee cup", "restaurant").
oa_rel("is", "coffee maker", "electric").
oa_rel("is", "dishwasher", "electric").
oa_rel("is used for", "closet", "storing clothes").
oa_rel("can", "water", "be liquid, ice, or steam").
oa_rel("is used for", "temple", "praying").
oa_rel("is used for", "carpet", "covering ugly floor").
oa_rel("is a sub-event of", "cleaning house", "vacuuming carpet").
oa_rel("can", "airplane", "land in field").
oa_rel("is used for", "closet", "hanging clothes").
oa_rel("can", "dog", "guide blind").
oa_rel("can", "umbrella", "shade you from sun").
oa_rel("is used for", "pot", "making soup").
oa_rel("is used for", "sidewalk", "riding bike on").
oa_rel("is", "fax machine", "electric").
oa_rel("is used for", "keyboard", "coding").
oa_rel("is", "sheep", "herbivorous").
oa_rel("is used for", "pen", "drawing").
oa_rel("usually appears in", "mouse", "office").
oa_rel("is used for", "dining room", "eating").
oa_rel("usually appears in", "bowl", "restaurant").
oa_rel("has", "orange", "vitamin C").
oa_rel("is used for", "sheet", "sleeping on").
oa_rel("is used for", "fork", "eating pie").
oa_rel("requires", "making pizza", "salt").
oa_rel("is", "knife", "dangerous").
oa_rel("is used for", "knife", "cutting food").
oa_rel("is used for", "fork", "lifting food from plate to mouth").
oa_rel("usually appears in", "saucer", "dining room").
oa_rel("is", "cream", "fluid").
oa_rel("requires", "beating egg", "spoon").
oa_rel("usually appears in", "straw", "restaurant").
oa_rel("is used for", "salt", "salting food").
oa_rel("usually appears in", "placemat", "restaurant").
oa_rel("is made from", "pasta", "flour").
oa_rel("requires", "baking cake", "egg").
oa_rel("is", "salt", "salty").
oa_rel("has", "strawberry", "vitamin C").
oa_rel("is made from", "butter", "milk").
oa_rel("is a sub-event of", "baking cake", "mixing butter with sugar").
oa_rel("is", "oatmeal", "fluid").
oa_rel("is made from", "juice", "fruit").
oa_rel("is made from", "casserole", "flour").
oa_rel("requires", "cutting", "knife").
oa_rel("has", "rice", "starch").
oa_rel("requires", "chopping", "knife").
oa_rel("usually appears in", "stool", "bar").
oa_rel("is used for", "refrigerator", "making ice").
oa_rel("can", "oven", "heat meal").
oa_rel("is a sub-event of", "baking bread", "preheating oven").
oa_rel("can", "boat", "sail through sea").
oa_rel("is used for", "scissors", "cutting string").
oa_rel("requires", "mincing", "scissors").
oa_rel("is used for", "living room", "watching tv").
oa_rel("usually appears in", "couch", "living room").
oa_rel("is used for", "stove", "frying burgers").
oa_rel("is used for", "ballon", "decoration").
oa_rel("is used for", "drum", "banging").
oa_rel("is", "beer", "harmful").
oa_rel("can", "bicycle", "travel on road").
oa_rel("is used for", "computer", "doing mathematical calculations").
oa_rel("usually appears in", "chair", "living room").
oa_rel("is used for", "projector", "showing presentations").
oa_rel("can", "computer", "save information").
oa_rel("is used for", "lobby", "meeting guests").
oa_rel("can", "fence", "divide property").
oa_rel("is used for", "container", "holding something").
oa_rel("is used for", "sailboat", "traveling on water").
oa_rel("can", "sailboat", "travel over water").
oa_rel("is used for", "mailbox", "sending letters").
oa_rel("is used for", "factory", "manufacture goods").
oa_rel("is used for", "broom", "sweeping floors").
oa_rel("requires", "cleaning house", "broom").
oa_rel("is used for", "scooter", "riding").
oa_rel("can", "airplanes", "carry people").
oa_rel("can", "hammer", "strike nail").
oa_rel("is used for", "van", "transporting people").
oa_rel("can", "refrigerator", "cool milk").
oa_rel("is used for", "bracelet", "decoration").
oa_rel("can", "ruler", "guide lines").
oa_rel("usually appears in", "dresser", "bedroom").
oa_rel("can", "snake", "eat egg").
oa_rel("requires", "beating egg", "mixer").
oa_rel("is used for", "spoon", "moving liquid food to mouth").
oa_rel("is used for", "spoon", "moving food to mouth").
oa_rel("is made from", "sandwich", "flour").
oa_rel("is used for", "elephant", "transporting people").
oa_rel("is used for", "house", "housing family").
oa_rel("can", "rat", "live in house").
oa_rel("is used for", "carpet", "protecting feet from floor").
oa_rel("is used for", "church", "worship").
oa_rel("can", "elephant", "weight 1000 kilos").
oa_rel("is used for", "fan", "cooling people").
oa_rel("is used for", "curtain", "get privacy").
oa_rel("is made from", "pizza slice", "flour").
oa_rel("is used for", "stapler", "stapling papers together").
oa_rel("usually appears in", "mug", "bar").
oa_rel("is used for", "boat", "keeping people out of water").
oa_rel("is made from", "chocolate", "cacao seeds").
oa_rel("can", "squirrel", "store nuts for winter").
oa_rel("is used for", "airplane", "carrying cargo").
oa_rel("usually appears in", "tractor", "farm").
oa_rel("can", "stove", "heat food").
oa_rel("has", "banana", "vitamin B").
oa_rel("is a sub-event of", "making juice", "cutting fruit").
oa_rel("is used for", "oven", "preparing food").
oa_rel("is used for", "museum", "displaying old objects").
oa_rel("is used for", "train station", "waiting train").
oa_rel("can", "frisbee", "descend slowly by hovering").
oa_rel("can", "cat", "eat meat").
oa_rel("can", "batter", "hit baseball").
oa_rel("is used for", "computer", "storing information").
oa_rel("can", "camera", "record scene").
oa_rel("usually appears in", "cattle", "farm").
oa_rel("can", "hammer", "nail board").
oa_rel("can", "bird", "spread wings").
oa_rel("is", "keyboard", "electric").
oa_rel("is used for", "stapler", "holding papers together").
oa_rel("can", "computer", "power down").
oa_rel("is used for", "thermometer", "measuring temperature").
oa_rel("is used for", "runway", "landing airplanes").
oa_rel("has", "zebra", "stripes").
oa_rel("can", "baseball bat", "hit baseball").
oa_rel("usually appears in", "hair dryer", "bathroom").
oa_rel("can", "ambulance", "travel on road").
oa_rel("is used for", "sofa", "sitting on").
oa_rel("can", "dog", "please human").
oa_rel("can", "raft", "travel over water").
oa_rel("is used for", "freeway", "driving car on").
oa_rel("is a sub-event of", "cleaning clothing", "hanging clothing up").
oa_rel("is used for", "lighthouse", "signaling danger").
oa_rel("is used for", "airplane", "travelling long distances").
oa_rel("can", "monkeys", "climb tree").
oa_rel("is used for", "train", "carrying cargo").
oa_rel("is used for", "printer", "printing pictures").
oa_rel("is used for", "printer", "printing books").
oa_rel("is used for", "train", "travelling long distances").
oa_rel("can", "elephant", "life tree").
oa_rel("can", "cat", "wash itself").
oa_rel("can", "train", "arrive at city").
oa_rel("is used for", "library", "finding information").
oa_rel("can", "turtle", "live in house").
oa_rel("can", "pigeon", "fly").
oa_rel("can", "axe", "hurt people").
oa_rel("can", "cat", "please humans").
oa_rel("has", "hot chocolate", "caffeine").
oa_rel("is", "sponge", "soft").
oa_rel("usually appears in", "stapler", "office").
oa_rel("is used for", "scooter", "transporting people").
oa_rel("usually appears in", "telephone", "office").
oa_rel("can", "toaster", "brown toast").
oa_rel("can", "bird", "lay eggs").
oa_rel("is used for", "mailbox", "receiving packages").
oa_rel("is used for", "restaurant", "meeting friends").
oa_rel("can", "rabbit", "live in house").
oa_rel("usually appears in", "tv", "living room").
oa_rel("usually appears in", "straw", "dining room").
oa_rel("has", "pineapple", "vitamin C").
oa_rel("usually appears in", "potato", "salad").
oa_rel("is", "orange", "sour").
oa_rel("is used for", "stool", "resting").
oa_rel("is used for", "piano", "playing music").
oa_rel("can", "suv", "transport people").
oa_rel("is used for", "grill", "grilling hamburgers").
oa_rel("can", "ship", "near island").
oa_rel("is used for", "phones", "listening to music").
oa_rel("is", "refrigerator", "electric").
oa_rel("usually appears in", "guitar", "rock band").
oa_rel("requires", "making juice", "juicer").
oa_rel("is used for", "subway", "transporting people").
oa_rel("is used for", "train car", "transporting people").
oa_rel("is made from", "fries", "potato").
oa_rel("is", "chicken", "omnivorous").
oa_rel("is used for", "oven", "heating food").
oa_rel("is used for", "oven", "baking food").
oa_rel("usually appears in", "container", "kitchen").
oa_rel("is used for", "fork", "eating solid food").
oa_rel("requires", "making pizza", "vegetables").
oa_rel("is", "puppy", "soft").
oa_rel("can", "owl", "see at night").
oa_rel("can", "pilot", "fly airplane").
oa_rel("is used for", "keyboard", "typing letters onto windows").
oa_rel("can", "charger", "charge battery").
oa_rel("can", "computer", "stream video").
oa_rel("is used for", "dresser", "holding cloth").
oa_rel("usually appears in", "hay", "barn").
oa_rel("is used for", "mailbox", "receiving bills").
oa_rel("is used for", "frisbee", "entertainment").
oa_rel("usually appears in", "water glass", "dining room").
oa_rel("can", "train", "transport many people at once").
oa_rel("can", "horse", "jump over objects").
oa_rel("can", "dog", "follow its master").
oa_rel("usually appears in", "penguin", "ocean").
oa_rel("can", "bear", "eat most types of food").
oa_rel("is used for", "couch", "relaxing").
oa_rel("is used for", "fireplace", "getting warm").
oa_rel("usually appears in", "refrigerator", "kitchen").
oa_rel("is used for", "phone", "finding information").
oa_rel("is", "horse", "herbivorous").
oa_rel("requires", "making juice", "fruit").
oa_rel("has", "egg", "vitamin B").
oa_rel("is a sub-event of", "baking cake", "mixing egg with sugar").
oa_rel("usually appears in", "turkey", "thanksgiving").
oa_rel("is used for", "waste basket", "storing trash").
oa_rel("is used for", "cake", "celebrating birthday").
oa_rel("is used for", "cake", "eating").
oa_rel("usually appears in", "tissue", "bathroom").
oa_rel("is used for", "speaker", "producing sound").
oa_rel("usually appears in", "bear", "jungle").
oa_rel("has", "monkey", "two legs").
oa_rel("usually appears in", "mat", "living room").
oa_rel("is used for", "fork", "piercing food").
oa_rel("is", "juice", "liquid").
oa_rel("can", "kitten", "live in house").
oa_rel("is used for", "barn", "storing farming equipment").
oa_rel("is used for", "chopsticks", "moving food to mouth").
oa_rel("is", "soup", "fluid").
oa_rel("is used for", "fruit stand", "buying and selling fruit").
oa_rel("can", "horse", "carry riders").
oa_rel("usually appears in", "curtain", "bedroom").
oa_rel("is used for", "dog", "herding sheep").
oa_rel("can", "cat", "be companion").
oa_rel("usually appears in", "dining table", "dining room").
oa_rel("is made from", "bun", "flour").
oa_rel("has", "egg", "iron").
oa_rel("is", "microwave", "electric").
oa_rel("can", "bear", "hunt rabbit").
oa_rel("is", "juice", "fluid").
oa_rel("requires", "mixing", "mixing bowl").
oa_rel("is used for", "cutting board", "cutting food").
oa_rel("requires", "baking cake", "butter").
oa_rel("is used for", "bread", "making toast").
oa_rel("has", "beans", "starch").
oa_rel("is made from", "toast", "flour").
oa_rel("usually appears in", "mousepad", "office").
oa_rel("can", "dog", "dig holes in yard").
oa_rel("has", "nut", "vitamin B").
oa_rel("requires", "frying", "frying pan").
oa_rel("is made from", "macaroni", "flour").
oa_rel("usually appears in", "glass", "dining room").
oa_rel("is used for", "bathroom", "clean humans").
oa_rel("is used for", "frisbee", "throwing").
oa_rel("usually appears in", "bedspread", "bedroom").
oa_rel("is used for", "tongs", "grasping food").
oa_rel("is used for", "vegetable", "eating").
oa_rel("requires", "mixing", "spoon").
oa_rel("has", "tomato", "vitamin C").
oa_rel("is", "peach", "sweet").
oa_rel("is a sub-event of", "making pizza", "baking pizza in oven").
oa_rel("is", "heater", "electric").
oa_rel("is used for", "guitar", "playing chords").
oa_rel("can", "jellyfish", "hurt person").
oa_rel("is used for", "ipod", "listening to music").
oa_rel("can", "vacuum", "clean carpet").
oa_rel("is used for", "harbor", "store boats").
oa_rel("is", "butter", "sticky").
oa_rel("can", "refrigerator", "cool warm food").
oa_rel("is used for", "binder", "holding papers together").
oa_rel("is", "vegetables", "healthy").
oa_rel("usually appears in", "napkin", "restaurant").
oa_rel("is", "beer", "liquid").
oa_rel("has", "pancake", "starch").
oa_rel("can", "frisbee", "fly").
oa_rel("is used for", "bathroom", "washing up").
oa_rel("is used for", "restaurant", "eating").
oa_rel("can", "van", "spend gas").
oa_rel("can", "rain", "wet clothes").
oa_rel("is used for", "ship", "transportation at sea").
oa_rel("is", "ice cream", "sweet").
oa_rel("is made from", "cake", "flour").
oa_rel("can", "school bus", "travel on road").
oa_rel("can", "airplane", "circle airfield").
oa_rel("usually appears in", "tablecloth", "restaurant").
oa_rel("is used for", "airplane", "transporting lots of people").
oa_rel("can", "turtle", "hide in its shell").
oa_rel("can", "oven", "brown chicken").
oa_rel("usually appears in", "nightstand", "bedroom").
oa_rel("usually appears in", "bowl", "dining room").
oa_rel("is used for", "pencil", "drawing").
oa_rel("has", "cheese", "calcium").
oa_rel("is a sub-event of", "cleaning house", "vacuuming floors").
oa_rel("can", "cat", "sleep most of day").
oa_rel("usually appears in", "faucet", "bathroom").
oa_rel("has", "soda", "water").
oa_rel("is used for", "bell", "making noise").
oa_rel("usually appears in", "hair clip", "bathroom").
oa_rel("can", "dishwasher", "wash dirty dishes").
oa_rel("is used for", "shop", "buying and selling").
oa_rel("usually appears in", "rolling pin", "kitchen").
oa_rel("usually appears in", "toaster oven", "kitchen").
oa_rel("usually appears in", "waiter", "restaurant").
oa_rel("usually appears in", "lobster", "water").
oa_rel("has", "lemon", "vitamin C").
oa_rel("is used for", "condiment", "flavoring food").
oa_rel("can", "ice maker", "making ice").
oa_rel("has", "beer", "water").
oa_rel("usually appears in", "wine glass", "dining room").
oa_rel("can", "computer", "boot from hard drive").
oa_rel("can", "cell phone", "ring").
oa_rel("is used for", "rug", "prevent scratches on floor").
oa_rel("usually appears in", "keyboard", "office").
oa_rel("is used for", "hair dryer", "dry hair").
oa_rel("is used for", "apartment", "housing family").
oa_rel("usually appears in", "cutting board", "kitchen").
oa_rel("is made from", "pizza pie", "flour").
oa_rel("usually appears in", "shower curtain", "bathroom").
oa_rel("is used for", "pantry", "keeping food organized").
oa_rel("has", "elephant", "long nose").
oa_rel("is used for", "spoon", "people to eat soup with").
oa_rel("is made from", "bagel", "flour").
oa_rel("usually appears in", "bill", "restaurant").
oa_rel("can", "knife", "cut potato").
oa_rel("has", "mushroom", "vitamin D").
oa_rel("can", "computer", "speed up research").
oa_rel("usually appears in", "pumpkin", "halloween").
oa_rel("usually appears in", "salt shaker", "restaurant").
oa_rel("can", "van", "carry few persons").
oa_rel("usually appears in", "pen", "office").
oa_rel("usually appears in", "coffee mug", "restaurant").
oa_rel("has", "kiwi", "vitamin C").
oa_rel("is", "pear", "sweet").
oa_rel("can", "grain", "provide complex carbohydrates").
oa_rel("is used for", "screw", "attaching item to something else").
oa_rel("is used for", "computer", "playing games").
oa_rel("usually appears in", "coffee mug", "dining room").
oa_rel("is used for", "soap", "washing clothes").
oa_rel("is used for", "bookshelf", "storing novels").
oa_rel("is used for", "hotel", "temporary residence").
oa_rel("is used for", "computer", "surfing internet").
oa_rel("usually appears in", "planter", "farm").
oa_rel("is", "desert", "dangerous").
oa_rel("can", "bird", "attempt to fly").
oa_rel("is used for", "air conditioner", "cooling air").
oa_rel("usually appears in", "piano", "orchestra").
oa_rel("requires", "making pizza", "cheese").
oa_rel("is", "blender", "electric").
oa_rel("is", "cliff", "dangerous").
oa_rel("is used for", "restaurant", "purchasing meals").
oa_rel("is", "soda", "fluid").
oa_rel("is used for", "mailbox", "storing mail").
oa_rel("can", "knife", "cut cheese").
oa_rel("requires", "crushing", "knife").
oa_rel("usually appears in", "binder", "office").
oa_rel("usually appears in", "toilet paper", "bathroom").
oa_rel("usually appears in", "sheet", "bedroom").
oa_rel("can", "air conditioner", "cool air").
oa_rel("is", "sauce", "fluid").
oa_rel("can", "gas stove", "heat food").
oa_rel("usually appears in", "toilet", "bathroom").
oa_rel("can", "refrigerator", "keep ice cold").
oa_rel("usually appears in", "pillow", "bedroom").
oa_rel("is used for", "dining room", "drinking").
oa_rel("is used for", "bread", "eating").
oa_rel("is a sub-event of", "juicing", "cutting fruit in half").
oa_rel("usually appears in", "fish", "water").
oa_rel("can", "oven", "bake").
oa_rel("is used for", "oven", "cooking").
oa_rel("usually appears in", "saucer", "restaurant").
oa_rel("is used for", "office", "holding meeting").
oa_rel("is made from", "beer", "hops").
oa_rel("can", "computer", "process information").
oa_rel("is used for", "comb", "removing tangles from hair").
oa_rel("requires", "washing dishes", "water").
oa_rel("is used for", "office", "working").
oa_rel("usually appears in", "tap", "bathroom").
oa_rel("usually appears in", "foil", "kitchen").
oa_rel("can", "dog", "be companion").
oa_rel("is used for", "raft", "transporting people").
oa_rel("can", "helmets", "prevent head injuries").
oa_rel("is", "sword", "dangerous").
oa_rel("has", "beans", "iron").
oa_rel("is", "pepper", "spicy").
oa_rel("is used for", "elephant", "riding").
oa_rel("is used for", "toothbrush", "cleaning teeth").
oa_rel("has", "broccoli", "vitamin C").
oa_rel("can", "rice", "provide complex carbohydrates").
oa_rel("can", "dog", "learn to fetch things").
oa_rel("can be", "steak", "cut").
oa_rel("is made from", "onion ring", "onion").
oa_rel("has", "cheese", "vitamin D").
oa_rel("is a sub-event of", "making juice", "running ingredients through juicer").
oa_rel("is used for", "spoon", "eating liquids").
oa_rel("usually appears in", "knife", "kitchen").
oa_rel("is", "pony", "herbivorous").
oa_rel("can", "grain", "provide energy").
oa_rel("is a sub-event of", "making coffee", "brewing coffee").
oa_rel("has", "pastry", "starch").
oa_rel("usually appears in", "mat", "bathroom").
oa_rel("is used for", "teddy bear", "having fun").
oa_rel("is made from", "cream", "milk").
oa_rel("has", "blueberry", "vitamin C").
oa_rel("is", "juice", "healthy").
oa_rel("can be", "chicken", "roasted").
oa_rel("can", "waiter", "serve food").
oa_rel("can", "bird", "be pet").
oa_rel("is used for", "spoon", "drinking").
oa_rel("can", "airplane", "fly").
oa_rel("is used for", "sailboat", "transportation at sea").
oa_rel("is", "lamb", "herbivorous").
oa_rel("is used for", "barn", "feeding animals").
oa_rel("usually appears in", "fork", "restaurant").
oa_rel("is made from", "bread", "flour").
oa_rel("has", "spinach", "vitamin B").
oa_rel("requires", "making pizza", "meat").
oa_rel("can", "cat", "be pet").
oa_rel("is used for", "baseball", "hitting").
oa_rel("requires", "coring", "knife").
oa_rel("is used for", "comb", "styling hair").
oa_rel("can", "computer", "stream media").
oa_rel("can", "suv", "carry few persons").
oa_rel("requires", "cleaning clothing", "water").
oa_rel("can", "cat", "jump onto table or chair").
oa_rel("is", "strawberry", "sweet").
oa_rel("can", "catcher", "catch").
oa_rel("is", "dog", "omnivorous").
oa_rel("is used for", "toaster", "toasting bread").
oa_rel("can be", "drinks", "drunk").
oa_rel("can", "police", "carry gun while at work").
oa_rel("is made from", "wine", "grapes").
oa_rel("is", "cat", "carnivorous").
oa_rel("is a sub-event of", "cleaning house", "polishing furniture").
oa_rel("can", "bird", "fly").
oa_rel("is", "eagle", "carnivorous").
oa_rel("is used for", "rug", "walking on").
oa_rel("is", "sweet potato", "sweet").
oa_rel("is used for", "oven", "roasting").
oa_rel("is used for", "dog", "providing friendship").
oa_rel("is", "gravy", "sticky").
oa_rel("is used for", "grill", "grilling steak").
oa_rel("can", "bear", "stand on their hind legs").
oa_rel("is used for", "farm", "raising crops").
oa_rel("can", "cow", "supply humans with milk").
oa_rel("is made from", "biscuit", "flour").
oa_rel("usually appears in", "seagull", "ocean").
oa_rel("is", "chili", "spicy").
oa_rel("usually appears in", "toaster", "kitchen").
oa_rel("can", "bread", "provide complex carbohydrates").
oa_rel("is used for", "lipstick", "coloring lips").
oa_rel("can", "computer", "run programs").
oa_rel("is", "broth", "fluid").
oa_rel("has", "broccoli", "calcium").
oa_rel("is used for", "baseball", "pitching").
oa_rel("is used for", "apple", "making juice").
oa_rel("has", "toast", "starch").
oa_rel("usually appears in", "tv stand", "living room").
oa_rel("can be", "chicken", "fried").
oa_rel("can", "dog", "come to its master").
oa_rel("can", "frog", "spring out of pond").
oa_rel("is used for", "computer", "sending email").
oa_rel("is a sub-event of", "juicing", "pressing halves on juicer").
oa_rel("is used for", "air conditioner", "cooling people").
oa_rel("is used for", "scissors", "cutting paper or cloth").
oa_rel("can", "dog", "sleep long time").
oa_rel("usually appears in", "spatula", "kitchen").
oa_rel("has", "alcohol", "water").
oa_rel("is used for", "bank", "saving money").
oa_rel("can", "bird", "fly high").
oa_rel("can", "screwdriver", "turn screw").
oa_rel("is", "cat", "soft").
oa_rel("is used for", "meat", "getting protein").
oa_rel("is used for", "canoe", "floating and moving on water").
oa_rel("is used for", "tunnel", "transportation").
oa_rel("has", "sweet potato", "vitamin C").
oa_rel("is", "giraffe", "herbivorous").
oa_rel("is used for", "screw", "fastening two objects together").
oa_rel("has", "tea", "water").
oa_rel("has", "tea", "caffeine").
oa_rel("can", "cat", "mind getting wet").
oa_rel("can", "jeep", "spend gas").
oa_rel("is made from", "noodles", "flour").
oa_rel("is used for", "sugar", "adding taste to food").
oa_rel("is used for", "cow", "milking").
oa_rel("is", "butter", "fluid").
oa_rel("usually appears in", "knife", "dining room").
oa_rel("is", "coffee", "liquid").
oa_rel("is", "coffee", "fluid").
oa_rel("is", "milk", "fluid").
oa_rel("can", "airplane", "travel through many time zones").
oa_rel("is used for", "canoe", "keeping people out of water").
oa_rel("has", "giraffe", "long neck").
oa_rel("has", "beef", "iron").
oa_rel("can", "ship", "go across sea").
oa_rel("can", "seagull", "fly").
oa_rel("is used for", "baseball field", "playing baseball with team").
oa_rel("can", "hats", "go on hat rack").
oa_rel("requires", "cooking", "food").
oa_rel("is", "oil", "liquid").
oa_rel("requires", "sauteing", "oil").
oa_rel("is", "toaster", "electric").
oa_rel("can", "computer", "mine data").
oa_rel("can", "bird", "chirp").
oa_rel("is", "cow", "herbivorous").
oa_rel("can", "knife", "be both tools and weapons").
oa_rel("usually appears in", "chef", "restaurant").
oa_rel("can be", "clothing", "hung").
oa_rel("is", "milk", "nutritious").
oa_rel("usually appears in", "bookshelf", "bedroom").
oa_rel("is", "cream", "sticky").
oa_rel("is made from", "burrito", "flour").
oa_rel("is a sub-event of", "baking bread", "gathering ingredients").
oa_rel("has", "coffee", "caffeine").
oa_rel("can be", "cardigan", "hung").
oa_rel("can", "bear", "swim").
oa_rel("is used for", "tattoo", "decoration").
oa_rel("can be", "pizza", "eaten").
oa_rel("can", "bird", "feed worms to its young").
oa_rel("is used for", "museum", "displaying historical artifacts").
oa_rel("can be", "pizza", "cut").
oa_rel("usually appears in", "kettle", "kitchen").
oa_rel("is used for", "milk", "feeding baby").
oa_rel("is", "air conditioner", "electric").
oa_rel("usually appears in", "duck", "water").
oa_rel("is used for", "mailbox", "sending packages").
oa_rel("requires", "grating", "grater").
oa_rel("has", "wine", "water").
oa_rel("is", "milk", "liquid").
oa_rel("can", "horse", "finish race").
oa_rel("usually appears in", "vegetable", "dinner").
oa_rel("is made from", "cupcake", "flour").
oa_rel("is", "polar bear", "carnivorous").
oa_rel("is used for", "pickup", "transporting goods").
oa_rel("is used for", "hotel", "staying overnight").
oa_rel("is used for", "headphones", "listening to music").
oa_rel("can", "pony", "be pet").
oa_rel("can", "horse", "jump over hurdles").
oa_rel("has", "peacock", "large").
oa_rel("is used for", "ship", "traveling on water").
oa_rel("is used for", "ship", "carrying cargo").
oa_rel("usually appears in", "ladle", "kitchen").
oa_rel("has", "beans", "calcium").
oa_rel("is", "marshmallow", "sweet").
oa_rel("usually appears in", "pepper shaker", "restaurant").
oa_rel("usually appears in", "pencil", "office").
oa_rel("is", "soda", "liquid").
oa_rel("is made from", "lemonade", "lemon").
oa_rel("is", "lemonade", "liquid").
oa_rel("can", "dog", "smell drugs").
oa_rel("usually appears in", "blade", "kitchen").
oa_rel("is", "lemon", "bitter").
oa_rel("is used for", "soap", "washing hands").
oa_rel("is a sub-event of", "baking cake", "pouring batter in cake pan").
oa_rel("is made from", "pizza", "flour").
oa_rel("is made from", "coffee", "coffee beans").
oa_rel("is used for", "temple", "worship").
oa_rel("is used for", "ship", "floating and moving on water").
oa_rel("has", "cheese", "vitamin B").
oa_rel("is made from", "pancake", "flour").
oa_rel("can", "cat", "hunt mice").
oa_rel("can", "owl", "hear slightest rustle").
oa_rel("usually appears in", "giraffe", "grassland").
oa_rel("usually appears in", "urinal", "bathroom").
oa_rel("has", "egg", "vitamin D").
oa_rel("is used for", "pencil", "writing").
oa_rel("usually appears in", "carpet", "bedroom").
oa_rel("requires", "juicing", "juicer").
oa_rel("is", "coffee", "bitter").
oa_rel("can", "owl", "fly").
oa_rel("is used for", "office", "conducting business").
oa_rel("can", "screw", "hold things together").
oa_rel("can", "police", "arrest").
oa_rel("has", "spinach", "iron").
oa_rel("has", "beef", "vitamin B").
oa_rel("can", "squirrel", "store nuts").
oa_rel("requires", "making pizza", "pizza pan").
oa_rel("requires", "coring", "slicer").
oa_rel("can", "thermometer", "measure temperature").
oa_rel("is made from", "cookie", "flour").
oa_rel("is", "beer", "fluid").
oa_rel("has", "pepper", "vitamin C").
oa_rel("is", "gun", "dangerous").
oa_rel("can", "soldier", "fight battle").
oa_rel("is used for", "apartment", "dwelling").
oa_rel("has", "spinach", "vitamin C").
oa_rel("usually appears in", "pizza tray", "dining room").
oa_rel("is used for", "vacuum", "cleaning carpet").
oa_rel("is a sub-event of", "cleaning house", "getting vacuum out").
oa_rel("is used for", "vending machine", "buying drinks").
oa_rel("is used for", "ground", "standing on").
oa_rel("is", "dip", "fluid").
oa_rel("can", "bird", "learn to fly").
oa_rel("usually appears in", "sheep", "meadow").
oa_rel("usually appears in", "turkey", "christmas").
oa_rel("usually appears in", "bread", "dinner").
oa_rel("has", "monkey", "two arms").
oa_rel("has", "milk", "water").
oa_rel("is", "honey", "sweet").
oa_rel("is", "lemon", "sour").
oa_rel("usually appears in", "fork", "dining room").
oa_rel("is used for", "highway", "transportation").
oa_rel("has", "almond", "vitamin B").
oa_rel("is made from", "yogurt", "milk").
oa_rel("can", "waitress", "serve food").
oa_rel("can", "pickup", "travel on road").
oa_rel("can", "bird", "build nest").
oa_rel("can", "jeep", "transport people").
oa_rel("is used for", "sheep", "shearing").
oa_rel("is used for", "library", "reading books").
oa_rel("can", "banjo", "play bluegrass music").
oa_rel("is used for", "sugar", "making drinks sweet").
oa_rel("usually appears in", "crab", "ocean").
oa_rel("is used for", "toothbrush", "keeping you teeth clean").
oa_rel("requires", "cleaning clothing", "washing machine").
oa_rel("usually appears in", "lobster", "ocean").
oa_rel("is", "game controller", "electric").
oa_rel("is used for", "doll", "entertainment").
oa_rel("is used for", "computer", "doing calculation").
oa_rel("can", "horse", "run faster than most humans").
oa_rel("is used for", "drum", "banging out rhythms").
oa_rel("has", "spinach", "vitamin D").
oa_rel("is used for", "kettle", "heating water").
oa_rel("can", "snail", "wave their antennae").
oa_rel("is used for", "oil", "frying food in").
oa_rel("usually appears in", "baking sheet", "kitchen").
oa_rel("usually appears in", "whale", "water").
oa_rel("can", "whale", "swim").
oa_rel("is", "dip", "sticky").
oa_rel("has", "juice", "vitamin C").
oa_rel("can", "canoe", "travel over water").
oa_rel("can", "eagle", "fly").
oa_rel("has", "sweet potato", "calcium").
oa_rel("is used for", "museum", "preserving historical artifacts").
oa_rel("is made from", "omelette", "flour").
oa_rel("is used for", "toaster", "making toast").
oa_rel("usually appears in", "cake", "birthday party").
oa_rel("has", "rice", "vitamin B").
oa_rel("is", "gravy", "fluid").
oa_rel("usually appears in", "wedding cake", "wedding").
oa_rel("is used for", "kettle", "making tea").
oa_rel("usually appears in", "dolphin", "ocean").
oa_rel("is", "shark", "dangerous").
oa_rel("is", "ipod", "electric").
oa_rel("usually appears in", "notepad", "office").
oa_rel("usually appears in", "tablecloth", "dining room").
oa_rel("can be", "lemon", "squeezed").
oa_rel("can", "horse", "be raced and ridden by humans").
oa_rel("has", "fish", "vitamin D").
oa_rel("usually appears in", "speaker", "office").
oa_rel("is used for", "ship", "travelling long distances").
oa_rel("can", "wine", "be ingredient in recipe").
oa_rel("requires", "baking bread", "dough").
oa_rel("is used for", "classroom", "teaching").
oa_rel("requires", "processing", "food processor").
oa_rel("has", "chocolate", "caffeine").
oa_rel("can", "cat", "sense with their whiskers").
oa_rel("is made from", "tortilla", "flour").
oa_rel("can", "fly", "fly").
oa_rel("can", "bat", "fly").
oa_rel("is", "tea", "fluid").
oa_rel("is used for", "axe", "chopping wood").
oa_rel("is used for", "double decker", "transporting lots of people").
oa_rel("is", "sugar", "sweet").
oa_rel("has", "raspberry", "vitamin C").
oa_rel("is", "monkey", "omnivorous").
oa_rel("usually appears in", "student", "classroom").
oa_rel("requires", "making pizza", "olive oil").
oa_rel("is used for", "knife", "slicing").
oa_rel("has", "tiger", "stripes").
oa_rel("usually appears in", "pencil sharpener", "office").
oa_rel("can", "dog", "sense danger").
oa_rel("can", "shuttle", "fly").
oa_rel("is used for", "condiment", "adding taste to food").
oa_rel("can be", "lemon", "eaten").
oa_rel("has", "corn", "starch").
oa_rel("can", "cat", "purr").
oa_rel("is used for", "mailbox", "receiving letters").
oa_rel("is used for", "frisbee", "having fun").
oa_rel("can", "police", "direct traffic").
oa_rel("usually appears in", "goat", "barn").
oa_rel("usually appears in", "grill", "kitchen").
oa_rel("is used for", "sausage", "getting protein").
oa_rel("is used for", "cafe", "having snack").
oa_rel("usually appears in", "zebra", "grassland").
oa_rel("is", "pig", "omnivorous").
oa_rel("has", "cranberry", "vitamin C").
oa_rel("can", "suv", "move quickly").
oa_rel("can", "frog", "catch fly").
oa_rel("usually appears in", "wine", "bar").
oa_rel("is used for", "factory", "manufacture things").
oa_rel("is", "olive oil", "liquid").
oa_rel("requires", "making pizza", "mixing bowl").
oa_rel("is used for", "sugar", "imparting specific flavor").
oa_rel("is made from", "pita", "flour").
oa_rel("is", "banana", "healthy").
oa_rel("can", "laptop", "save files on disk").
oa_rel("usually appears in", "hand soap", "bathroom").
oa_rel("is used for", "air conditioner", "lowering air temperature").
oa_rel("is used for", "bus", "transportation").
oa_rel("can be", "jeans", "hung").
oa_rel("is used for", "teddy bear", "entertainment").
oa_rel("is used for", "supermarket", "buying and selling").
oa_rel("can", "dog", "be pet").
oa_rel("is made from", "burger", "flour").
oa_rel("is made from", "hamburger", "flour").
oa_rel("is made from", "tea", "leaves of camellia sinensis").
oa_rel("is", "peanut butter", "fluid").
oa_rel("can", "subway", "transport many people at once").
oa_rel("usually appears in", "toothpaste", "bathroom").
oa_rel("is used for", "sailboat", "keeping people out of water").
oa_rel("is a sub-event of", "making pizza", "rising dough").
oa_rel("can", "monkeys", "throw things").
oa_rel("is used for", "backyard", "planting flowers").
oa_rel("is a sub-event of", "baking bread", "rising dough").
oa_rel("requires", "cooking", "heat").
oa_rel("can", "shark", "swim").
oa_rel("can", "frog", "catch flies with its tongue").
oa_rel("is used for", "computer", "enjoyment").
oa_rel("is used for", "double decker", "transporting people").
oa_rel("usually appears in", "chalk", "classroom").
oa_rel("can", "flamingo", "fly").
oa_rel("requires", "peeling", "peeler").
oa_rel("is used for", "condiment", "imparting specific flavor").
oa_rel("usually appears in", "pork", "dinner").
oa_rel("usually appears in", "knife block", "kitchen").
oa_rel("is made from", "cheesecake", "flour").
oa_rel("has", "lime", "vitamin C").
oa_rel("can", "ship", "travel over water").
oa_rel("can", "bird", "perch").
oa_rel("has", "wheat", "starch").
oa_rel("is a sub-event of", "baking bread", "put batter in oven").
oa_rel("usually appears in", "groom", "wedding").
oa_rel("is", "video camera", "electric").
oa_rel("is", "cream", "sweet").
oa_rel("is used for", "cafe", "meeting people").
oa_rel("is used for", "freeway", "transportation").
oa_rel("is used for", "toothpaste", "cleaning teeth").
oa_rel("usually appears in", "teacher", "classroom").
oa_rel("can", "dove", "fly").
oa_rel("is used for", "laptop", "studying").
oa_rel("has", "cereal", "starch").
oa_rel("can be", "jacket", "hung").
oa_rel("usually appears in", "monkey", "jungle").
oa_rel("is used for", "scooter", "transportation").
oa_rel("is", "toothpaste", "sticky").
oa_rel("is", "lotion", "fluid").
oa_rel("usually appears in", "chopsticks", "restaurant").
oa_rel("can", "hammer", "nail nail").
oa_rel("is used for", "soap", "bathing").
oa_rel("can be", "food", "cut").
oa_rel("has", "mango", "vitamin C").
oa_rel("is", "calf", "herbivorous").
oa_rel("is", "cola", "fluid").
oa_rel("usually appears in", "spoon", "restaurant").
oa_rel("is used for", "drum", "making rhythm").
oa_rel("is used for", "sugar", "flavoring food").
oa_rel("is used for", "salt", "imparting specific flavor").
oa_rel("can", "sedan", "transport people").
oa_rel("usually appears in", "drinks", "dinner").
oa_rel("is used for", "comb", "combing hair").
oa_rel("is", "donkey", "herbivorous").
oa_rel("usually appears in", "tongs", "kitchen").
oa_rel("is used for", "hairbrush", "styling hair").
oa_rel("usually appears in", "comb", "bathroom").
oa_rel("usually appears in", "tiger", "jungle").
oa_rel("usually appears in", "mill", "barn").
oa_rel("is a sub-event of", "baking cake", "gathering ingredients").
oa_rel("can", "duck", "fly").
oa_rel("can be", "duck", "roasted").
oa_rel("usually appears in", "swan", "water").
oa_rel("can", "minivan", "carry few persons").
oa_rel("usually appears in", "santa", "christmas").
oa_rel("is used for", "couch", "resting").
oa_rel("is", "peanut butter", "sticky").
oa_rel("requires", "mixing", "electric mixer").
oa_rel("can", "duck", "swim").
oa_rel("usually appears in", "pizza tray", "restaurant").
oa_rel("is made from", "whipped cream", "milk").
oa_rel("is", "guacamole", "fluid").
oa_rel("can", "cat", "jump amazingly high").
oa_rel("can", "swan", "fly").
oa_rel("is used for", "bread", "eating").
oa_rel("can be", "toast", "toasted").
oa_rel("usually appears in", "menu", "bar").
oa_rel("can", "parrot", "imitate human voices").
oa_rel("has", "milk", "calcium").
oa_rel("has", "cappuccino", "caffeine").
oa_rel("is", "caramel", "fluid").
oa_rel("usually appears in", "computer", "office").
oa_rel("is used for", "attic", "storing books").
oa_rel("is used for", "phone", "listening to music").
oa_rel("is", "hard drive", "electric").
oa_rel("can be", "wine", "drunk").
oa_rel("can", "kitten", "be pet").
oa_rel("is used for", "cafe", "meeting friends").
oa_rel("usually appears in", "wii", "living room").
oa_rel("can", "flamingo", "be pet").
oa_rel("is made from", "loaf", "flour").
oa_rel("is used for", "laptop", "doing calculation").
oa_rel("has", "milkshake", "water").
oa_rel("can", "alligator", "swim").
oa_rel("can", "laptop", "process information").
oa_rel("can", "swan", "swim").
oa_rel("is used for", "library", "borrowing books").
oa_rel("usually appears in", "knife", "restaurant").
oa_rel("is used for", "truck", "transporting people").
oa_rel("is used for", "truck", "transportation").
oa_rel("can", "goose", "swim").
oa_rel("is", "ice cream", "fluid").
oa_rel("requires", "baking bread", "flour").
oa_rel("has", "champagne", "alcohol").
oa_rel("is used for", "salt", "flavoring food").
oa_rel("usually appears in", "goose", "water").
oa_rel("usually appears in", "ostrich", "grassland").
oa_rel("is used for", "camel", "transporting people").
oa_rel("is used for", "grill", "cooking foods").
oa_rel("is used for", "laptop", "finding information").
oa_rel("is", "deer", "herbivorous").
oa_rel("is used for", "backyard", "growing garden").
oa_rel("can", "frog", "jump very high").
oa_rel("can", "parrot", "fly").
oa_rel("is used for", "drum", "hitting").
oa_rel("has", "pepperoni", "vitamin B").
oa_rel("is", "milk", "healthy").
oa_rel("is used for", "mall", "meeting friends").
oa_rel("can", "double decker", "travel on road").
oa_rel("can", "van", "move quickly").
oa_rel("is", "spear", "dangerous").
oa_rel("usually appears in", "alligator", "water").
oa_rel("has", "tofu", "iron").
oa_rel("is used for", "chicken", "getting protein").
oa_rel("is used for", "motorcycle", "transportation").
oa_rel("is used for", "grill", "barbecuing foods").
oa_rel("usually appears in", "whisk", "kitchen").
oa_rel("is used for", "hotel", "sleeping").
oa_rel("is used for", "laptop", "playing games").
oa_rel("is used for", "hammer", "pounding in nails").
oa_rel("is used for", "shoe", "protecting feet").
oa_rel("usually appears in", "toiletries", "bathroom").
oa_rel("is used for", "restaurant", "eating meal without cooking").
oa_rel("requires", "cleaning house", "vacuum").
oa_rel("can", "bee", "fly").
oa_rel("is used for", "refrigerator", "storing foods").
oa_rel("is", "goat", "herbivorous").
oa_rel("requires", "baking cake", "cake pan").
oa_rel("is made from", "pretzel", "flour").
oa_rel("usually appears in", "alcohol", "bar").
oa_rel("is", "alcohol", "liquid").
oa_rel("is", "champagne", "fluid").
oa_rel("usually appears in", "dishes", "dining room").
oa_rel("is used for", "laptop", "doing mathematical calculations").
oa_rel("is", "owl", "carnivorous").
oa_rel("can", "suv", "spend gas").
oa_rel("usually appears in", "beer", "bar").
oa_rel("can", "toaster", "brown bread").
oa_rel("is used for", "water", "drinking").
oa_rel("is used for", "screwdriver", "inserting screw").
oa_rel("is used for", "salt", "adding taste to food").
oa_rel("usually appears in", "octopus", "ocean").
oa_rel("is", "blood", "liquid").
oa_rel("has", "yogurt", "calcium").
oa_rel("can", "helicopter", "fly").
oa_rel("usually appears in", "computer desk", "office").
oa_rel("is made from", "waffle", "flour").
oa_rel("is made from", "brownie", "flour").
oa_rel("is used for", "lotion", "moisturizing skin").
oa_rel("has", "bagel", "starch").
oa_rel("has", "peanut butter", "iron").
oa_rel("is", "food processor", "electric").
oa_rel("is", "olive oil", "fluid").
oa_rel("can", "hummingbird", "fly").
oa_rel("usually appears in", "pepper shaker", "dining room").
oa_rel("can", "frog", "be pet").
oa_rel("is used for", "hotel", "staying on vacations").
oa_rel("is used for", "train", "transportation").
oa_rel("usually appears in", "waitress", "restaurant").
oa_rel("is used for", "metal stand", "sitting on").
oa_rel("requires", "coring", "corer").
oa_rel("can", "van", "transport people").
oa_rel("is", "champagne", "liquid").
oa_rel("is used for", "backyard", "growing vegetables").
oa_rel("can", "beaker", "measure liquid").
oa_rel("is", "shampoo", "sticky").
oa_rel("is", "moose", "herbivorous").
oa_rel("requires", "baking cake", "dough").
oa_rel("is", "whipped cream", "sweet").
oa_rel("can", "gas stove", "heat eater").
oa_rel("usually appears in", "octopus", "water").
oa_rel("is used for", "bicycle", "transportation").
oa_rel("is", "kitten", "soft").
oa_rel("can", "bartender", "mix classic cocktails").
oa_rel("is", "cattle", "herbivorous").
oa_rel("can", "cell phone", "make call").
oa_rel("can", "can opener", "open cans").
oa_rel("is", "rice cooker", "electric").
oa_rel("is used for", "pantry", "storing food").
oa_rel("is used for", "dishwasher", "washing dishes").
oa_rel("is", "peach", "healthy").
oa_rel("has", "peacock", "elaborate plumage").
oa_rel("can be", "juice", "drunk").
oa_rel("can", "rabbit", "be pet").
oa_rel("is used for", "wine", "drinking").
oa_rel("requires", "making coffee", "coffee beans").
oa_rel("is used for", "drum", "playing music").
oa_rel("is a sub-event of", "baking bread", "kneading dough").
oa_rel("can", "police", "tail suspect").
oa_rel("is used for", "suv", "transporting people").
oa_rel("is used for", "pickup", "transporting handful of people").
oa_rel("is", "milkshake", "liquid").
oa_rel("is", "router", "electric").
oa_rel("is a sub-event of", "making pizza", "shaping dough into thin circle").
oa_rel("is", "bull", "herbivorous").
oa_rel("usually appears in", "shark", "water").
oa_rel("is used for", "stuffed animal", "entertainment").
oa_rel("is", "hawk", "carnivorous").
oa_rel("is used for", "sailboat", "transporting people").
oa_rel("is", "shaving cream", "sticky").
oa_rel("is used for", "banana", "eating").
oa_rel("can", "chicken", "sing").
oa_rel("is used for", "mitt", "protecting hand").
oa_rel("usually appears in", "crab", "water").
oa_rel("is", "lion", "carnivorous").
oa_rel("usually appears in", "water glass", "restaurant").
oa_rel("usually appears in", "dvd player", "living room").
oa_rel("is used for", "listening to music", "entertainment").
oa_rel("can", "robin", "winter down south").
oa_rel("is", "shampoo", "fluid").
oa_rel("is", "oil", "fluid").
oa_rel("is", "tiger", "carnivorous").
oa_rel("is", "bomb", "dangerous").
oa_rel("can", "cereal", "provide complex carbohydrates").
oa_rel("usually appears in", "printer", "office").
oa_rel("has", "alcohol", "alcohol").
oa_rel("is used for", "hairbrush", "combing hair").
oa_rel("has", "almond", "calcium").
oa_rel("is", "lion", "dangerous").
oa_rel("is used for", "projector", "showing films").
oa_rel("usually appears in", "champagne", "wedding").
oa_rel("is", "milk", "good for baby").
oa_rel("usually appears in", "wok", "kitchen").
oa_rel("is used for", "sugar", "sweetening coffee").
oa_rel("usually appears in", "dish drainer", "kitchen").
oa_rel("usually appears in", "baking pan", "kitchen").
oa_rel("is", "alcohol", "fluid").
oa_rel("is used for", "ruler", "measuring lengths").
oa_rel("is used for", "beer", "drinking").
oa_rel("can", "poodle", "be pet").
oa_rel("is used for", "cafe", "drinking coffee").
oa_rel("can", "theater", "show movie").
oa_rel("requires", "washing dishes", "dish cloth").
oa_rel("can", "puppy", "be pet").
oa_rel("usually appears in", "shampoo", "bathroom").
oa_rel("is", "blood", "fluid").
oa_rel("is", "toothpaste", "fluid").
oa_rel("usually appears in", "penguin", "water").
oa_rel("can", "heater", "warm feet").
oa_rel("is", "pear", "healthy").
oa_rel("is", "toaster oven", "electric").
oa_rel("is used for", "helicopter", "traversing skies").
oa_rel("is", "cooking oil", "fluid").
oa_rel("requires", "frying", "cooking oil").
oa_rel("is a sub-event of", "baking cake", "put cake pan in oven").
oa_rel("usually appears in", "bartender", "bar").
oa_rel("is", "guacamole", "sticky").
oa_rel("can", "seal", "position itself on rock").
oa_rel("is used for", "screwdriver", "installing or removing screws").
oa_rel("usually appears in", "glue stick", "office").
oa_rel("is made from", "egg roll", "flour").
oa_rel("usually appears in", "pizza pan", "kitchen").
oa_rel("is", "hummus", "fluid").
oa_rel("can be", "hot dog", "eaten").
oa_rel("usually appears in", "ruler", "office").
oa_rel("is used for", "donut", "eating").
oa_rel("has", "grains", "vitamin B").
oa_rel("is", "liquor", "liquid").
oa_rel("is", "liquor", "harmful").
oa_rel("is used for", "helicopter", "transporting people").
oa_rel("is used for", "backyard", "family activities").
oa_rel("is", "lotion", "sticky").
oa_rel("usually appears in", "staples", "office").
oa_rel("can", "goose", "fly").
oa_rel("is", "milkshake", "fluid").
oa_rel("can", "eagle", "feed worms to its young").
oa_rel("can", "peacock", "fly").
oa_rel("usually appears in", "bride", "wedding").
oa_rel("is a sub-event of", "mincing", "cutting ingredients into tinier pieces").
oa_rel("is", "cola", "liquid").
oa_rel("is", "rabbit", "herbivorous").
oa_rel("is used for", "backyard", "children to play in").
oa_rel("can", "penguin", "swim").
oa_rel("is", "hand dryer", "electric").
oa_rel("is", "kitten", "carnivorous").
oa_rel("has", "oyster", "iron").
oa_rel("is used for", "camel", "riding").
oa_rel("is", "gorilla", "omnivorous").
oa_rel("can", "rat", "be pet").
oa_rel("is used for", "truck", "transporting handful of people").
oa_rel("is used for", "instruments", "making music").
oa_rel("is", "lemonade", "fluid").
oa_rel("is used for", "backyard", "having barbecues").
oa_rel("usually appears in", "trumpet", "orchestra").
oa_rel("is used for", "laptop", "surfing internet").
oa_rel("has", "cappuccino", "water").
oa_rel("is", "caramel", "sticky").
oa_rel("has", "cheesecake", "starch").
oa_rel("is", "chocolate", "sweet").
oa_rel("can", "bicycle", "travel on road").
oa_rel("is used for", "backyard", "dogs to run around in").
oa_rel("is used for", "wine", "satisfying thirst").
oa_rel("is", "perfume", "liquid").
oa_rel("usually appears in", "beer mug", "bar").
oa_rel("can", "snake", "be pet").
oa_rel("is", "rifle", "dangerous").
oa_rel("can be", "coffee", "drunk").
oa_rel("has", "tofu", "calcium").
oa_rel("can", "air conditioner", "warm air").
oa_rel("is used for", "pantry", "storing kitchen items").
oa_rel("requires", "making pizza", "flour").
oa_rel("is", "strawberry", "healthy").
oa_rel("is used for", "laptop", "enjoyment").
oa_rel("has", "waffle", "starch").
oa_rel("usually appears in", "lion", "grassland").
oa_rel("can", "eagle", "spot prey from afar").
oa_rel("requires", "cleaning clothing", "detergent").
oa_rel("usually appears in", "dolphin", "water").
oa_rel("is", "pesto", "sticky").
oa_rel("can", "duck", "attempt to fly").
oa_rel("is", "tiger", "dangerous").
oa_rel("is used for", "backyard", "having party").
oa_rel("is", "lime", "healthy").
oa_rel("has", "lemonade", "water").
oa_rel("can", "wheat", "provide complex carbohydrates").
oa_rel("is", "yogurt", "fluid").
oa_rel("usually appears in", "frog", "water").
oa_rel("usually appears in", "bass", "rock band").
oa_rel("is used for", "couch", "lying down").
oa_rel("usually appears in", "lizard", "dessert").
oa_rel("has", "liquor", "water").
oa_rel("can", "jeep", "move quickly").
oa_rel("usually appears in", "shark", "ocean").
oa_rel("usually appears in", "mixing bowl", "kitchen").
oa_rel("is", "ice maker", "electric").
oa_rel("usually appears in", "hamburger", "lunch").
oa_rel("is made from", "cheeseburger", "flour").
oa_rel("is used for", "accordion", "polka music").
oa_rel("is used for", "hammer", "pulling out nails").
oa_rel("is a sub-event of", "baking cake", "stirring flour").
oa_rel("can be", "cake", "eaten").
oa_rel("has", "chicken", "vitamin B").
oa_rel("is used for", "water bottle", "holding drinks").
oa_rel("is", "gummy bear", "sweet").
oa_rel("is", "fast food", "unhealthy").
oa_rel("usually appears in", "ape", "jungle").
oa_rel("is used for", "lemon", "making juice").
oa_rel("usually appears in", "hairbrush", "bathroom").
oa_rel("can be", "water", "drunk").
oa_rel("is used for", "cleat", "protecting feet").
oa_rel("can", "penguin", "be pet").
oa_rel("usually appears in", "pizza oven", "kitchen").
oa_rel("has", "kangaroo", "two legs").
oa_rel("is used for", "perfume", "aroma").
oa_rel("requires", "baking cake", "flour").
oa_rel("can be", "tea", "drunk").
oa_rel("usually appears in", "whale", "ocean").
oa_rel("is", "cappuccino", "fluid").
oa_rel("is", "cappuccino", "liquid").
oa_rel("is", "washing machine", "electric").
oa_rel("can", "turtle", "be pet").
oa_rel("can", "frog", "swim").
oa_rel("is used for", "subway", "transporting lots of people").
oa_rel("usually appears in", "champagne", "bar").
oa_rel("usually appears in", "skillet", "kitchen").
oa_rel("is", "cotton candy", "sweet").
oa_rel("usually appears in", "otter", "water").
oa_rel("can", "robin", "fly").
oa_rel("is used for", "boat", "transportation").
oa_rel("is used for", "crane", "lifting heavy weight").
oa_rel("usually appears in", "grater", "kitchen").
oa_rel("is used for", "trumpet", "playing music").
oa_rel("is used for", "cafeteria", "eating").
oa_rel("is", "yogurt", "sweet").
oa_rel("is used for", "mall", "shopping").
oa_rel("is used for", "boot", "protecting feet").
oa_rel("can", "laptop", "run programs").
oa_rel("can be", "vest", "hung").
oa_rel("is used for", "helmet", "protecting head").
oa_rel("is", "vinegar", "liquid").
oa_rel("is used for", "beans", "eating").
oa_rel("can", "lizard", "sun to warm up").
oa_rel("is used for", "milk", "satisfying thirst").
oa_rel("can", "pickup", "spend gas").
oa_rel("can", "hawk", "fly").
oa_rel("can", "police", "tail criminal").
oa_rel("is", "fast food", "cheap").
oa_rel("can", "violin", "play beautiful music").
oa_rel("usually appears in", "eagle", "jungle").
oa_rel("usually appears in", "beer mug", "restaurant").
oa_rel("is used for", "hairbrush", "removing tangles from hair").
oa_rel("is", "seal", "carnivorous").
oa_rel("is used for", "factory", "making products").
oa_rel("usually appears in", "chopsticks", "dining room").
oa_rel("is", "pudding", "sweet").
oa_rel("is used for", "shampoo", "washing hair").
oa_rel("has", "oatmeal", "starch").
oa_rel("has", "kangaroo", "two arms").
oa_rel("can", "jeep", "carry few persons").
oa_rel("can", "lizard", "be pet").
oa_rel("is used for", "attic", "storing dishes").
oa_rel("is", "liquor", "fluid").
oa_rel("can", "minivan", "spend gas").
oa_rel("usually appears in", "fax machine", "office").
oa_rel("can be", "polo shirt", "hung").
oa_rel("is", "pizza oven", "electric").
oa_rel("has", "gorilla", "two legs").
oa_rel("can", "minivan", "transport people").
oa_rel("can", "snake", "hurt").
oa_rel("requires", "making pizza", "yeast").
oa_rel("can", "snake", "be dangerous").
oa_rel("usually appears in", "cake pan", "kitchen").
oa_rel("usually appears in", "shaving cream", "bathroom").
oa_rel("is", "crow", "omnivorous").
oa_rel("usually appears in", "turtle", "water").
oa_rel("is used for", "hairbrush", "brushing hair").
oa_rel("is used for", "sneaker", "protecting feet").
oa_rel("can", "rice cooker", "cooking rice").
oa_rel("is used for", "attic", "storing things that are not used").
oa_rel("can", "hen", "fly").
oa_rel("is used for", "water", "satisfying thirst").
oa_rel("can", "camel", "work for days without water").
oa_rel("has", "champagne", "water").
oa_rel("is made from", "milkshake", "milk").
oa_rel("can", "crow", "fly").
oa_rel("is used for", "beer", "satisfying thirst").
oa_rel("can", "parrot", "be pet").
oa_rel("is used for", "shampoo", "curing dandruff").
oa_rel("is used for", "screwdriver", "fixing loose screws").
oa_rel("is used for", "jet", "transporting people").
oa_rel("is used for", "guitar", "making music").
oa_rel("can", "sedan", "spend gas").
oa_rel("is used for", "sedan", "transporting people").
oa_rel("is used for", "attic", "keeping stuff in").
oa_rel("is", "pesto", "fluid").
oa_rel("is used for", "violin", "playing music").
oa_rel("can be", "soda", "drunk").
oa_rel("can", "dolphin", "swim").
oa_rel("is", "dark chocolate", "bitter").
oa_rel("is used for", "cereal", "eating").
oa_rel("has", "liquor", "alcohol").
oa_rel("is", "wolf", "dangerous").
oa_rel("has", "steak", "vitamin B").
oa_rel("can", "bartender", "serve alcoholic or soft drink beverages").
oa_rel("is used for", "drum", "making music").
oa_rel("is made from", "bread loaf", "flour").
oa_rel("is", "alpaca", "herbivorous").
oa_rel("is", "vacuum", "electric").
oa_rel("is used for", "fork", "moving food").
oa_rel("is used for", "mall", "buying and selling").
oa_rel("can", "chicken", "fly").
oa_rel("usually appears in", "electric toothbrush", "bathroom").
oa_rel("usually appears in", "leopard", "jungle").
oa_rel("usually appears in", "utensil holder", "kitchen").
oa_rel("usually appears in", "drum", "orchestra").
oa_rel("requires", "cooking", "ingredients").
oa_rel("can", "fork", "lift food").
oa_rel("can", "laptop", "save information").
oa_rel("is used for", "baseball glove", "protecting hand").
oa_rel("is used for", "attic", "storing clothes").
oa_rel("can", "bald eagle", "fly").
oa_rel("usually appears in", "banquet table", "restaurant").
oa_rel("has", "bacon", "vitamin B").
oa_rel("is used for", "laptop", "sending email").
oa_rel("has", "ape", "two legs").
oa_rel("can", "chicken", "spread wings").
oa_rel("is", "shaving cream", "fluid").
oa_rel("can", "duck", "lay eggs").
oa_rel("has", "gorilla", "two arms").
oa_rel("can be", "shorts", "hung").
oa_rel("is", "vinegar", "fluid").
oa_rel("can be", "beer", "drunk").
oa_rel("is used for", "accordion", "playing music").
oa_rel("can be", "pants", "hung").
oa_rel("can", "pickup", "move quickly").
oa_rel("is made from", "oreo", "flour").
oa_rel("is", "rabbit", "soft").
oa_rel("is", "silk", "soft").
oa_rel("is used for", "laptop", "data storage").
oa_rel("can be", "milk", "drunk").
oa_rel("usually appears in", "drum", "rock band").
oa_rel("usually appears in", "wedding gown", "wedding").
oa_rel("is used for", "soda", "satisfying thirst").
oa_rel("is used for", "sandal", "protecting feet").
oa_rel("is", "hippo", "herbivorous").
oa_rel("is used for", "cafe", "eating cookies").
oa_rel("can be", "skirt", "hung").
oa_rel("can", "turtle", "live to be 200 years old").
oa_rel("can", "laptop", "speed up research").
oa_rel("has", "chicken liver", "iron").
oa_rel("has", "beef liver", "iron").
oa_rel("has", "rabbit", "two long ears").
oa_rel("has", "ape", "two arms").
oa_rel("requires", "cleaning clothing", "washing powders").
oa_rel("requires", "cleaning house", "mop").
oa_rel("requires", "baking bread", "yeast").
oa_rel("requires", "baking bread", "bread pan").
oa_rel("requires", "beating egg", "electric mixer").
oa_rel("is a sub-event of", "making pizza", "mixing flour, yeast, olive oil").
oa_rel("is a sub-event of", "making pizza", "add sauce, cheese, meats, vegetables").
oa_rel("is a sub-event of", "making coffee", "grinding coffee beans").
oa_rel("can be", "", "diced").
oa_rel("can be", "", "sliced").
oa_rel("can be", "", "minced").
oa_rel("can be", "", "shred").
oa_rel("can be", "", "baked").
oa_rel("is made of", "sushi", "rice").
oa_rel("is made from", "whey", "milk").
oa_rel("is created by", "", "baking").
oa_rel("is", "fast food", "bad for health").
oa_rel("is", "doughnut", "sweet").
oa_rel("is", "fudge", "sweet").
oa_rel("is", "milk chocolate", "sweet").
oa_rel("is", "cooking oil", "liquid").
oa_rel("is", "hummus", "sticky").
oa_rel("is", "leopard", "dangerous").
oa_rel("is", "cheetah", "dangerous").
oa_rel("is", "bison", "herbivorous").
oa_rel("is", "ape", "omnivorous").
oa_rel("is", "cheetah", "carnivorous").
oa_rel("is", "raccoon", "carnivorous").
oa_rel("is", "otter", "carnivorous").
oa_rel("is", "copier", "electric").
oa_rel("is used for", "fork", "piercing solid food").
oa_rel("is used for", "fork", "moving solid food").
oa_rel("is used for", "spoon", "eating soft food").
oa_rel("is used for", "spoon", "moving soft food to mouth").
oa_rel("is used for", "spoon", "moving soft food").
oa_rel("is used for", "spoon", "scooping soft food").
oa_rel("is capable of", "microwave", "heating food").
oa_rel("is used for", "harp", "playing music").
oa_rel("can be used for", "drum", "playing in orchestra").
oa_rel("can be used for", "wood", "campfires").
oa_rel("is used for", "perfume", "perfuming").
oa_rel("can", "steering wheel", "control vehicle").
oa_rel("can", "steering wheel", "control direction of vehicle").
oa_rel("can", "finch", "fly").
oa_rel("can", "firefly", "fly").
oa_rel("can", "snake", "bite").
oa_rel("can", "lizard", "sun itself on rock").
oa_rel("can", "turtle", "swim").
oa_rel("can", "otter", "swim").
oa_rel("can", "starch", "provide complex carbohydrates").
oa_rel("can", "starch", "provide energy").
oa_rel("is good at", "chef", "cook food").
oa_rel("is good at", "chef", "prepare food").
oa_rel("is good at", "chef", "season food").
oa_rel("is good at", "chef", "cook fish").
oa_rel("is good at", "chef", "season meat").
oa_rel("can", "catcher", "catch baseball").
oa_rel("can", "snowplows", "clear snow from roads").
oa_rel("can", "cell phone", "vibrate").
oa_rel("can", "helmet", "prevent head injuries").
oa_rel("can", "spoon", "scoop food").
oa_rel("usually appears in", "main course", "dinner").
oa_rel("usually appears in", "side dishes", "dinner").
oa_rel("usually appears in", "fast food", "lunch").
oa_rel("usually appears in", "soft drinks", "lunch").
oa_rel("usually appears in", "violin", "orchestra").
oa_rel("usually appears in", "clarinet", "orchestra").
oa_rel("usually appears in", "trumpet", "brass band").
oa_rel("usually appears in", "gas stove", "kitchen").
oa_rel("usually appears in", "cooking pot", "kitchen").
oa_rel("usually appears in", "beer mug", "dining room").
oa_rel("usually appears in", "barkeeper", "bar").
oa_rel("usually appears in", "cocktail cabinet", "bar").
oa_rel("usually appears in", "liquor", "bar").
oa_rel("usually appears in", "restaurant table", "restaurant").
oa_rel("usually appears in", "copier", "office").
oa_rel("usually appears in", "camel", "dessert").
oa_rel("usually appears in", "jaguar", "jungle").
oa_rel("usually appears in", "gorilla", "jungle").
oa_rel("usually appears in", "bison", "grassland").
oa_rel("usually appears in", "rhino", "grassland").
oa_rel("usually appears in", "moose", "meadow").
is_a(person_type_01, person).
is_a(person_type_02, person).
is_a(person_type_03, person).
is_a(person_type_04, person).
is_a(person_type_05, person).
is_a(man, person_type_01).
is_a(woman, person_type_01).
is_a(lady, person_type_02).
is_a(gentleman, person_type_02).
is_a(boy, person_type_03).
is_a(girl, person_type_03).
is_a(baby, person_type_04).
is_a(toddler, person_type_04).
is_a(child, person_type_04).
is_a(teenager, person_type_04).
is_a(adult, person_type_04).
is_a(elder, person_type_04).
is_a(pedestrian, person_type_05).
is_a(passenger, person_type_05).
is_a(spectator, person_type_05).
is_a(tourist, person_type_05).
is_a(spectators, person_type_05).
is_a(customers, person_type_05).
is_a(visitor, person_type_05).

'''

def npp_sub(match):

    #print(match.group(0))
    #print(match.group(0).replace(" ", "_").replace('"','').replace("-",""))
    return match.group(0).replace(" ", "_").replace('"','').replace("-","_").replace(",","")


KG_REL = KG_REL.replace('", "','"; "').lower()
KG_REL = re.sub(r"\"[a-z \-,]*\"", npp_sub, KG_REL).replace(";",",").replace(", ,",", ")
KG_FACTS= KG_FACTS.replace('", "','"; "').lower()
KG_FACTS = re.sub(r"\"[a-z \-,]*\"", npp_sub, KG_FACTS).replace(";",",").replace(", ,",", ")

KG = KG_REL + KG_FACTS


RULES_old = '''is_a(A, C) :- is_a(A, B), is_a(B, C).
is_a(N, "thing") :- name(N, T).
name(N1, Oid) :- name(N2, Oid), is_a(N2, N1).

oa_rel("is a type of", Obj, Attr) :- is_a(Obj, Attr).
oa_rel(R, Obj, Attr) :- in_oa_rel(R, Obj, Attr).
oa_rel(R, Obj, Attr) :- is_a(Obj, T), oa_rel(R, T, Attr).'''


# RULES_OLD = '''
# %sixRULES
# is_a(A, C) :- is_a(A, B), is_a(B, C). 
# is_a(N, thing) :- name(N, T).
# name(N1, Oid) :- name(N2, Oid), is_a(N2, N1).

# oa_rel(is_a_type_of, Obj, Attr) :- is_a(Obj, Attr).
# oa_rel(R, Obj, Attr) :- in_oa_rel(R, Obj, Attr).
# oa_rel(R, Obj, Attr) :- is_a(Obj, T), oa_rel(R, T, Attr).
# '''

RULES = '''
%sixRULES
is_a(A, C) :- is_a(A, B), is_a(B, C). 
is_a(N, thing) :- name(T,N).
name(Oid,N1) :- name(Oid,N2), is_a(N2,N1).

oa_rel(is_a_type_of, Obj, Attr) :- is_a(Obj, Attr).
%oa_rel(R, Obj, Attr) :- in_oa_rel(R, Obj, Attr).
oa_rel(R, Obj, Attr) :- is_a(Obj, T), oa_rel(R, T, Attr).
'''





"""
['window', 'man', 'shirt', 'tree', 'wall', 'person', 'building', 'ground', 'sky', 'sign', 'head', 'pole', 'hand', 'grass', 'hair', 'leg', 'car', 'woman', 'leaves', 'table', 'trees', 'ear', 'pants', 'people', 'eye', 'door', 'water', 'fence', 'wheel', 'nose', 'chair', 'floor', 'arm', 'jacket', 'hat', 'shoe', 'tail', 'face', 'leaf', 'clouds', 'number', 'letter', 'plate', 'windows', 'shorts', 'road', 'sidewalk', 'flower', 'bag', 'helmet', 'snow', 'rock', 'boy', 'tire', 'logo', 'cloud', 'roof', 'glass', 'street', 'foot', 'legs', 'umbrella', 'post', 'jeans', 'mouth', 'boat', 'cap', 'bottle', 'girl', 'bush', 'shoes', 'flowers', 'glasses', 'field', 'picture', 'mirror', 'bench', 'box', 'bird', 'dirt', 'clock', 'neck', 'food', 'letters', 'bowl', 'shelf', 'bus', 'train', 'pillow', 'trunk', 'horse', 'plant', 'coat', 'airplane', 'lamp', 'wing', 'kite', 'elephant', 'paper', 'seat', 'dog', 'cup', 'house', 'counter', 'sheep', 'street light', 'glove', 'banana', 'branch', 'giraffe', 'rocks', 'cow', 'book', 'truck', 'racket', 'flag', 'ceiling', 'skateboard', 'cabinet', 'eyes', 'ball', 'zebra', 'bike', 'wheels', 'sand', 'hands', 'surfboard', 'frame', 'feet', 'windshield', 'finger', 'motorcycle', 'player', 'bushes', 'hill', 'child', 'bed', 'cat', 'sink', 'container', 'sock', 'tie', 'towel', 'traffic light', 'pizza', 'paw', 'backpack', 'collar', 'mountain', 'lid', 'basket', 'vase', 'phone', 'animal', 'sticker', 'branches', 'donut', 'lady', 'mane', 'license plate', 'cheese', 'fur', 'laptop', 'uniform', 'wire', 'fork', 'beach', 'wrist', 'buildings', 'word', 'desk', 'toilet', 'cars', 'curtain', 'pot', 'bear', 'ears', 'tag', 'dress', 'tower', 'faucet', 'screen', 'cell phone', 'watch', 'keyboard', 'arrow', 'sneakers', 'stone', 'blanket', 'broccoli', 'orange', 'numbers', 'drawer', 'knife', 'fruit', 'ocean', 't-shirt', 'cord', 'guy', 'spots', 'apple', 'napkin', 'cone', 'bread', 'bananas', 'sweater', 'cake', 'bicycle', 'skis', 'vehicle', 'room', 'couch', 'frisbee', 'horn', 'air', 'plants', 'trash can', 'camera', 'paint', 'ski', 'tomato', 'tiles', 'belt', 'words', 'television', 'wires', 'tray', 'socks', 'pipe', 'bat', 'rope', 'bathroom', 'carrot', 'suit', 'books', 'boot', 'sauce', 'ring', 'spoon', 'bricks', 'meat', 'van', 'bridge', 'goggles', 'platform', 'gravel', 'vest', 'label', 'stick', 'pavement', 'beak', 'refrigerator', 'computer', 'wetsuit', 'mountains', 'gloves', 'balcony', 'tree trunk', 'carpet', 'skirt', 'palm tree', 'fire hydrant', 'chain', 'kitchen', 'jersey', 'candle', 'remote control', 'shore', 'boots', 'rug', 'suitcase', 'computer mouse', 'clothes', 'street sign', 'pocket', 'outlet', 'can', 'snowboard', 'net', 'horns', 'pepper', 'doors', 'stairs', 'scarf', 'gate', 'graffiti', 'purse', 'luggage', 'beard', 'vegetables', 'bracelet', 'necklace', 'wristband', 'parking lot', 'park', 'train tracks', 'onion', 'dish', 'statue', 'sun', 'vegetable', 'sandwich', 'arms', 'star', 'doorway', 'wine', 'path', 'skier', 'teeth', 'men', 'stove', 'crust', 'weeds', 'chimney', 'chairs', 'feathers', 'monitor', 'home plate', 'speaker', 'fingers', 'steps', 'catcher', 'trash', 'forest', 'blinds', 'log', 'bathtub', 'eye glasses', 'outfit', 'cabinets', 'countertop', 'lettuce', 'pine tree', 'oven', 'city', 'walkway', 'jar', 'cart', 'tent', 'curtains', 'painting', 'dock', 'cockpit', 'step', 'pen', 'poster', 'light switch', 'hot dog', 'frosting', 'bucket', 'bun', 'pepperoni', 'crowd', 'thumb', 'propeller', 'symbol', 'liquid', 'tablecloth', 'pan', 'runway', 'train car', 'teddy bear', 'baby', 'microwave', 'headband', 'earring', 'store', 'spectator', 'fan', 'headboard', 'straw', 'stop sign', 'skin', 'pillows', 'display', 'couple', 'ladder', 'bottles', 'sprinkles', 'power lines', 'tennis ball', 'papers', 'decoration', 'burner', 'birds', 'umpire', 'grill', 'brush', 'cable', 'tongue', 'smoke', 'canopy', 'wings', 'controller', 'carrots', 'river', 'taxi', 'drain', 'spectators', 'sheet', 'game', 'chicken', 'american flag', 'mask', 'stones', 'batter', 'topping', 'wine glass', 'suv', 'tank top', 'scissors', 'barrier', 'hills', 'olive', 'planter', 'animals', 'plates', 'cross', 'mug', 'baseball', 'boats', 'telephone pole', 'mat', 'crosswalk', 'lips', 'mushroom', 'hay', 'toilet paper', 'tape', 'hillside', 'surfer', 'apples', 'donuts', 'station', 'mustache', 'children', 'pond', 'tires', 'toy', 'walls', 'flags', 'boxes', 'sofa', 'tomatoes', 'bags', 'sandals', 'onions', 'sandal', 'oranges', 'paws', 'pitcher', 'houses', 'baseball bat', 'moss', 'duck', 'apron', 'airport', 'light bulb', 'drawers', 'toothbrush', 'shelves', 'potato', 'light fixture', 'umbrellas', 'drink', 'heart', 'fence post', 'egg', 'hose', 'power line', 'fireplace', 'icing', 'nightstand', 'vehicles', 'magnet', 'beer', 'hook', 'comforter', 'lake', 'bookshelf', 'fries', 'peppers', 'coffee table', 'sweatshirt', 'shower', 'jet', 'water bottle', 'cows', 'entrance', 'driver', 'towels', 'soap', 'sail', 'crate', 'utensil', 'salad', 'kites', 'paddle', 'mound', 'tree branch']
['white', 'black', 'green', 'blue', 'brown', 'red', 'gray', 'large', 'small', 'wooden', 'yellow', 'tall', 'metal', 'orange', 'long', 'dark', 'silver', 'pink', 'standing', 'clear', 'round', 'glass', 'open', 'sitting', 'short', 'parked', 'plastic', 'walking', 'brick', 'tan', 'purple', 'striped', 'colorful', 'cloudy', 'hanging', 'concrete', 'blond', 'bare', 'empty', 'young', 'old', 'closed', 'baseball', 'happy', 'bright', 'wet', 'gold', 'stone', 'smiling', 'light', 'dirty', 'flying', 'shiny', 'plaid', 'on', 'thin', 'square', 'tennis', 'little', 'sliced', 'leafy', 'playing', 'thick', 'beige', 'steel', 'calm', 'rectangular', 'dry', 'tiled', 'leather', 'eating', 'painted', 'ceramic', 'pointy', 'lying', 'surfing', 'snowy', 'paved', 'clean', 'fluffy', 'electric', 'cooked', 'grassy', 'stacked', 'full', 'covered', 'paper', 'framed', 'lit', 'blurry', 'grazing', 'flat', 'leafless', 'skiing', 'curved', 'light brown', 'beautiful', 'decorative', 'up', 'folded', 'sandy', 'chain-link', 'arched', 'overcast', 'cut', 'wide', 'running', 'waiting', 'ripe', 'long sleeved', 'furry', 'rusty', 'short sleeved', 'down', 'light blue', 'cloudless', 'dark brown', 'high', 'hazy', 'fresh', 'chocolate', 'cream colored', 'baby', 'worn', 'bent', 'light colored', 'rocky', 'skinny', 'curly', 'patterned', 'driving', 'jumping', 'maroon', 'riding', 'raised', 'lush', 'dark blue', 'off', 'cardboard', 'reflective', 'bald', 'iron', 'floral', 'black and white', 'melted', 'piled', 'skateboarding', 'rubber', 'talking', 'chrome', 'wire', 'puffy', 'broken', 'smooth', 'low', 'evergreen', 'narrow', 'denim', 'grouped', 'wicker', 'straight', 'triangular', 'sunny', 'dried', 'bushy', 'resting', 'sleeping', 'wrinkled', 'adult', 'dark colored', 'hairy', 'khaki', 'stuffed', 'wavy', 'nike', 'chopped', 'curled', 'shirtless', 'splashing', 'posing', 'upside down', 'ski', 'water', 'pointing', 'double decker', 'glazed', 'male', 'marble', 'fried', 'rock', 'ornate', 'wild', 'shining', 'tinted', 'asphalt', 'filled', 'floating', 'burnt', 'crossed', 'fuzzy', 'outdoors', 'overhead', 'potted', 'muddy', 'pale', 'decorated', 'swinging', 'asian', 'sharp', 'floppy', 'outstretched', 'rough', 'drinking', 'displayed', 'lined', 'having meeting', 'reflected', 'delicious', 'barefoot', 'plain', 'healthy', 'printed', 'frosted', 'crouching', 'written', 'illuminated', 'aluminum', 'digital', 'skating', 'trimmed', 'patchy', 'bending', 'soft', 'toasted', 'neon', 'choppy', 'fake', 'wispy', 'toy', 'knit', 'uncooked', 'vertical', 'tied', 'female', 'straw', 'grilled', 'rolled', 'rounded', 'wrapped', 'attached', 'messy', 'bathroom', 'swimming', 'deep', 'pretty', 'sleeveless', 'granite', 'rainbow colored', 'fallen', 'modern', 'kneeling', 'kitchen', 'turned', 'used', 'murky', 'busy', 'snowboarding', 'still', 'soccer', 'mounted', 'antique', 'street', 'watching', 'slanted', 'faded', 'glowing', 'teal', 'gravel', 'balding', 'outdoor', 'heavy', 'cracked', 'snow', 'baked', 'fancy', 'transparent', 'roman', 'vintage', 'old fashioned', 'cooking', 'horizontal', 'crouched', 'shredded', 'computer', 'docked', 'navy', 'octagonal', 'shallow', 'sparse', 'hard', 'cotton', 'fat', 'waving', 'carpeted', 'electronic', 'protective', 'foggy', 'rippled', 'cloth', 'squatting', 'shaggy', 'scattered', 'new', 'barren', 'stained', 'wireless', 'designed', 'reading', 'overgrown', 'apple', 'looking down', 'sliding', 'ocean', 'plush', 'tilted', 'shaded', 'dusty', 'flowered', 'crumpled', 'dotted', 'mesh', 'collared', 'traffic', 'woven', 'athletic', 'lighted', 'power', 'clay', 'made', 'crispy', 'elderly', 'bunched', 'warm', 'cluttered', 'brunette', 'textured', 'spread', 'brass', 'chinese', 'styrofoam', 'elevated', 'palm', 'blowing', 'copper', 'foamy', 'unripe', 'cheese', 'sheer', 'padded', 'half full', 'shaped', 'laughing', 'winter', 'manicured', 'indoors', 'speckled', 'batting', 'loose', 'caucasian', 'peeling', 'diced', 'dense', 'adidas', 'military', 'tin', 'halved', 'jeans', 'wired', 'mixed', 'bronze', 'alert', 'neat', 'public', 'drawn', 'cobblestone', 'massive', 'real', 'peeled', 'assorted', 'sunlit', 'chipped', 'blooming', 'american', 'perched', 'shaved', 'spiky', 'wool', 'bamboo', 'gloved', 'spraying', 'commercial', 'steep', 'braided', 'steamed', 'tight', 'partly cloudy', 'unpeeled', 'carved', 'damaged', 'edged', 'directional', 'strong', 'pulled back', 'wrinkly', 'muscular', 'tabby', 'mowed', 'disposable', 'portable', 'shut', 'looking up', 'sad', 'packed', 'rippling', 'vanilla', 'eaten', 'barbed', 'safety', 'shingled', 'roasted', 'juicy', 'pine', 'shadowed', 'diamond', 'crystal', 'angled', 'weathered', 'homemade', 'seasoned', 'paneled', 'sloped', 'sweet', 'miniature', 'domed', 'pizza', 'torn', 'crowded', 'tail', 'beaded', 'suspended', 'cushioned', 'tangled', 'stormy', 'blank', 'tasty', 'hitting', 'deciduous', 'frozen', 'license', 'fenced', 'sprinkled', 'feathered', 'dull', 'tropical', 'twisted', 'crooked', 'wine', 'browned', 'cylindrical', 'greasy', 'abandoned', 'capital', 'toilet', 'unlit', 'iced', 'upholstered', 'gloomy', 'professional', 'creamy', 'analog', 'jagged', 'oblong', 'ruffled', 'connected', 'ivory', 'christmas', 'upper', 'coffee', 'lace', 'ridged', 'comfortable', 'foreign', 'knotted', 'french', 'support', 'performing trick', 'artificial', 'spiral', 'crumbled', 'dangling', 'cordless', 'wedding', 'fire', 'rustic', 'trash', 'uneven', 'grated', 'curvy', 'telephone', 'unoccupied', 'curious', 'rotten', 'oriental', 'folding', 'crusty', 'rimmed', 'birthday', 'polished', 'funny', 'crashing', 'powdered', 'lower', 'winding', 'calico', 'banana', 'angry', 'inflated', 'scrambled', 'wii', 'vast', 'sleepy', 'rainy', 'groomed', 'glossy', 'wooded', 'sturdy', 'vibrant', 'misty', 'beer', 'office', 'melting', 'silk', 'discolored', 'corded', 'formal', 'vacant', 'oversized', 'vinyl', 'spinning', 'simple', 'translucent', 'mature', 'tomato', 'crisp', 'boiled', 'recessed', 'gas', 'handmade', 'clumped', 'railroad', 'rugged', 'sculpted', 'burning', 'patched', 'industrial', 'chubby', 'ugly', 'garbage', 'powerful', 'elongated', 'pepper', 'abundant', 'opaque', 'ornamental', 'fluorescent', 'soda', 'packaged', 'inflatable', 'unpaved', 'soap', 'strawberry', 'park', 'exterior', 'intricate', 'sealed', 'quilted', 'hollow', 'breakable', 'pastel', 'urban', 'wrist', 'breaking', 'regular', 'forested', 'abstract', 'unhappy', 'incomplete', 'fine', 'chalk', 'coarse', 'polo', 'unhealthy', 'irregular', 'polar', 'uncomfortable', 'typical', 'complete', 'scarce', 'immature']
['right', 'left', 'on', 'wearing', 'near', 'in', 'of', 'behind', 'in front of', 'under', 'holding', 'with', 'by', 'on the side of', 'watching', 'inside', 'carrying', 'eating', 'at', 'riding', 'playing', 'covered by', 'covering', 'touching', 'on the front of', 'full of', 'riding on', 'using', 'covered in', 'sitting at', 'playing with', 'crossing', 'on the back of', 'surrounded by', 'reflected in', 'covered with', 'flying', 'pulled by', 'contain', 'standing by', 'driving on', 'surrounding', 'walking down', 'printed on', 'attached to', 'talking on', 'facing', 'driving', 'worn on', 'floating in', 'grazing on', 'on the bottom of', 'standing in front of', 'topped with', 'playing in', 'walking with', 'swimming in', 'driving down', 'pushed by', 'playing on', 'close to', 'waiting for', 'between', 'running on', 'tied to', 'on the edge of', 'holding onto', 'talking to', 'eating from', 'perched on', 'parked by', 'reaching for', 'connected to', 'grazing in', 'floating on', 'wrapped around', 'skiing on', 'stacked on', 'parked at', 'standing at', 'walking across', 'plugged into', 'stuck on', 'stuck in', 'drinking from', 'seen through', 'sitting by', 'sitting in front of', 'looking out', 'parked in front of', 'petting', 'wrapped in', 'selling', 'parked along', 'coming from', 'lying inside', 'sitting inside', 'walking by', 'sitting with', 'making', 'walking through', 'hung on', 'walking along', 'going down', 'leaving', 'running in', 'flying through', 'mounted to', 'on the other side of', 'licking', 'sniffing', 'followed by', 'following', 'riding in', 'biting', 'parked alongside', 'chasing', 'leading', 'hanging off', 'helping', 'coming out of', 'hanging out of', 'staring at', 'walking toward', 'served on', 'hugging', 'entering', 'skiing in', 'looking in', 'draped over', 'tied around', 'exiting', 'looking down at', 'looking into', 'drawn on', 'balancing on', 'jumping over', 'posing with', 'reflecting in', 'eating at', 'walking up', 'sewn on', 'reflected on', 'about to hit', 'getting on', 'observing', 'approaching', 'traveling on', 'walking towards', 'wading in', 'growing by', 'displayed on', 'growing along', 'mixed with', 'grabbing', 'jumping on', 'scattered on', 'climbing', 'pointing at', 'opening', 'taller than', 'going into', 'decorated by', 'decorating', 'preparing', 'coming down', 'tossing', 'eating in', 'growing from', 'chewing', 'washing', 'herding', 'skiing down', 'looking through', 'picking up', 'trying to catch', 'standing against', 'looking over', 'typing on', 'piled on', 'tying', 'shining through', 'smoking', 'cleaning', 'walking to', 'smiling at', 'guiding', 'dragging', 'chained to', 'going through', 'enclosing', 'adjusting', 'running through', 'skating on', 'photographing', 'smelling', 'kissing', 'falling off', 'decorated with', 'walking into', 'worn around', 'walking past', 'blowing out', 'towing', 'jumping off', 'sprinkled on', 'moving', 'running across', 'hidden by', 'traveling down', 'splashing', 'hang from', 'looking toward', 'kept in', 'cooked in', 'displayed in', 'sitting atop', 'brushing', 'buying', 'in between', 'larger than', 'smaller than', 'pouring', 'playing at', 'longer than', 'higher than', 'jumping in', 'shorter than', 'bigger than']

"""

""" NAMES
0 window
1 man
2 shirt
3 tree
4 wall
5 person
6 building
7 ground
8 sky
9 sign
10 head
11 pole
12 hand
13 grass
14 hair
15 leg
16 car
17 woman
18 leaves
19 table
20 trees
21 ear
22 pants
23 people
24 eye
25 door
26 water
27 fence
28 wheel
29 nose
30 chair
31 floor
32 arm
33 jacket
34 hat
35 shoe
36 tail
37 face
38 leaf
39 clouds
40 number
41 letter
42 plate
43 windows
44 shorts
45 road
46 sidewalk
47 flower
48 bag
49 helmet
50 snow
51 rock
52 boy
53 tire
54 logo
55 cloud
56 roof
57 glass
58 street
59 foot
60 legs
61 umbrella
62 post
63 jeans
64 mouth
65 boat
66 cap
67 bottle
68 girl
69 bush
70 shoes
71 flowers
72 glasses
73 field
74 picture
75 mirror
76 bench
77 box
78 bird
79 dirt
80 clock
81 neck
82 food
83 letters
84 bowl
85 shelf
86 bus
87 train
88 pillow
89 trunk
90 horse
91 plant
92 coat
93 airplane
94 lamp
95 wing
96 kite
97 elephant
98 paper
99 seat
100 dog
101 cup
102 house
103 counter
104 sheep
105 street light
106 glove
107 banana
108 branch
109 giraffe
110 rocks
111 cow
112 book
113 truck
114 racket
115 flag
116 ceiling
117 skateboard
118 cabinet
119 eyes
120 ball
121 zebra
122 bike
123 wheels
124 sand
125 hands
126 surfboard
127 frame
128 feet
129 windshield
130 finger
131 motorcycle
132 player
133 bushes
134 hill
135 child
136 bed
137 cat
138 sink
139 container
140 sock
141 tie
142 towel
143 traffic light
144 pizza
145 paw
146 backpack
147 collar
148 mountain
149 lid
150 basket
151 vase
152 phone
153 animal
154 sticker
155 branches
156 donut
157 lady
158 mane
159 license plate
160 cheese
161 fur
162 laptop
163 uniform
164 wire
165 fork
166 beach
167 wrist
168 buildings
169 word
170 desk
171 toilet
172 cars
173 curtain
174 pot
175 bear
176 ears
177 tag
178 dress
179 tower
180 faucet
181 screen
182 cell phone
183 watch
184 keyboard
185 arrow
186 sneakers
187 stone
188 blanket
189 broccoli
190 orange
191 numbers
192 drawer
193 knife
194 fruit
195 ocean
196 t-shirt
197 cord
198 guy
199 spots
200 apple
201 napkin
202 cone
203 bread
204 bananas
205 sweater
206 cake
207 bicycle
208 skis
209 vehicle
210 room
211 couch
212 frisbee
213 horn
214 air
215 plants
216 trash can
217 camera
218 paint
219 ski
220 tomato
221 tiles
222 belt
223 words
224 television
225 wires
226 tray
227 socks
228 pipe
229 bat
230 rope
231 bathroom
232 carrot
233 suit
234 books
235 boot
236 sauce
237 ring
238 spoon
239 bricks
240 meat
241 van
242 bridge
243 goggles
244 platform
245 gravel
246 vest
247 label
248 stick
249 pavement
250 beak
251 refrigerator
252 computer
253 wetsuit
254 mountains
255 gloves
256 balcony
257 tree trunk
258 carpet
259 skirt
260 palm tree
261 fire hydrant
262 chain
263 kitchen
264 jersey
265 candle
266 remote control
267 shore
268 boots
269 rug
270 suitcase
271 computer mouse
272 clothes
273 street sign
274 pocket
275 outlet
276 can
277 snowboard
278 net
279 horns
280 pepper
281 doors
282 stairs
283 scarf
284 gate
285 graffiti
286 purse
287 luggage
288 beard
289 vegetables
290 bracelet
291 necklace
292 wristband
293 parking lot
294 park
295 train tracks
296 onion
297 dish
298 statue
299 sun
300 vegetable
301 sandwich
302 arms
303 star
304 doorway
305 wine
306 path
307 skier
308 teeth
309 men
310 stove
311 crust
312 weeds
313 chimney
314 chairs
315 feathers
316 monitor
317 home plate
318 speaker
319 fingers
320 steps
321 catcher
322 trash
323 forest
324 blinds
325 log
326 bathtub
327 eye glasses
328 outfit
329 cabinets
330 countertop
331 lettuce
332 pine tree
333 oven
334 city
335 walkway
336 jar
337 cart
338 tent
339 curtains
340 painting
341 dock
342 cockpit
343 step
344 pen
345 poster
346 light switch
347 hot dog
348 frosting
349 bucket
350 bun
351 pepperoni
352 crowd
353 thumb
354 propeller
355 symbol
356 liquid
357 tablecloth
358 pan
359 runway
360 train car
361 teddy bear
362 baby
363 microwave
364 headband
365 earring
366 store
367 spectator
368 fan
369 headboard
370 straw
371 stop sign
372 skin
373 pillows
374 display
375 couple
376 ladder
377 bottles
378 sprinkles
379 power lines
380 tennis ball
381 papers
382 decoration
383 burner
384 birds
385 umpire
386 grill
387 brush
388 cable
389 tongue
390 smoke
391 canopy
392 wings
393 controller
394 carrots
395 river
396 taxi
397 drain
398 spectators
399 sheet
400 game
401 chicken
402 american flag
403 mask
404 stones
405 batter
406 topping
407 wine glass
408 suv
409 tank top
410 scissors
411 barrier
412 hills
413 olive
414 planter
415 animals
416 plates
417 cross
418 mug
419 baseball
420 boats
421 telephone pole
422 mat
423 crosswalk
424 lips
425 mushroom
426 hay
427 toilet paper
428 tape
429 hillside
430 surfer
431 apples
432 donuts
433 station
434 mustache
435 children
436 pond
437 tires
438 toy
439 walls
440 flags
441 boxes
442 sofa
443 tomatoes
444 bags
445 sandals
446 onions
447 sandal
448 oranges
449 paws
450 pitcher
451 houses
452 baseball bat
453 moss
454 duck
455 apron
456 airport
457 light bulb
458 drawers
459 toothbrush
460 shelves
461 potato
462 light fixture
463 umbrellas
464 drink
465 heart
466 fence post
467 egg
468 hose
469 power line
470 fireplace
471 icing
472 nightstand
473 vehicles
474 magnet
475 beer
476 hook
477 comforter
478 lake
479 bookshelf
480 fries
481 peppers
482 coffee table
483 sweatshirt 
484 shower
485 jet
486 water bottle
487 cows
488 entrance
489 driver
490 towels
491 soap
492 sail
493 crate
494 utensil
495 salad
496 kites
497 paddle
498 mound
499 tree branch
"""

""" attributes
0 white
1 black
2 green
3 blue
4 brown
5 red
6 gray
7 large
8 small
9 wooden
10 yellow
11 tall
12 metal
13 orange
14 long
15 dark
16 silver
17 pink
18 standing
19 clear
20 round
21 glass
22 open
23 sitting
24 short
25 parked
26 plastic
27 walking
28 brick
29 tan
30 purple
31 striped
32 colorful
33 cloudy
34 hanging
35 concrete
36 blond
37 bare
38 empty
39 young
40 old
41 closed
42 baseball
43 happy
44 bright
45 wet
46 gold
47 stone
48 smiling
49 light
50 dirty
51 flying
52 shiny
53 plaid
54 on
55 thin
56 square
57 tennis
58 little
59 sliced
60 leafy
61 playing
62 thick
63 beige
64 steel
65 calm
66 rectangular
67 dry
68 tiled
69 leather
70 eating
71 painted
72 ceramic
73 pointy
74 lying
75 surfing
76 snowy
77 paved
78 clean
79 fluffy
80 electric
81 cooked
82 grassy
83 stacked
84 full
85 covered
86 paper
87 framed
88 lit
89 blurry
90 grazing
91 flat
92 leafless
93 skiing
94 curved
95 light brown
96 beautiful
97 decorative
98 up
99 folded
100 sandy
101 chain-link
102 arched
103 overcast
104 cut
105 wide
106 running
107 waiting
108 ripe
109 long sleeved
110 furry
111 rusty
112 short sleeved
113 down
114 light blue
115 cloudless
116 dark brown
117 high
118 hazy
119 fresh
120 chocolate
121 cream colored
122 baby
123 worn
124 bent
125 light colored
126 rocky
127 skinny
128 curly
129 patterned
130 driving
131 jumping
132 maroon
133 riding
134 raised
135 lush
136 dark blue
137 off
138 cardboard
139 reflective
140 bald
141 iron
142 floral
143 black and white
144 melted
145 piled
146 skateboarding
147 rubber
148 talking
149 chrome
150 wire
151 puffy
152 broken
153 smooth
154 low
155 evergreen
156 narrow
157 denim
158 grouped
159 wicker
160 straight
161 triangular
162 sunny
163 dried
164 bushy
165 resting
166 sleeping
167 wrinkled
168 adult
169 dark colored
170 hairy
171 khaki
172 stuffed
173 wavy
174 nike
175 chopped
176 curled
177 shirtless
178 splashing
179 posing
180 upside down
181 ski
182 water
183 pointing
184 double decker
185 glazed
186 male
187 marble
188 fried
189 rock
190 ornate
191 wild
192 shining
193 tinted
194 asphalt
195 filled
196 floating
197 burnt
198 crossed
199 fuzzy
200 outdoors
201 overhead
202 potted
203 muddy
204 pale
205 decorated
206 swinging
207 asian
208 sharp
209 floppy
210 outstretched
211 rough
212 drinking
213 displayed
214 lined
215 having meeting
216 reflected
217 delicious
218 barefoot
219 plain
220 healthy
221 printed
222 frosted
223 crouching
224 written
225 illuminated
226 aluminum
227 digital
228 skating
229 trimmed
230 patchy
231 bending
232 soft
233 toasted
234 neon
235 choppy
236 fake
237 wispy
238 toy
239 knit
240 uncooked
241 vertical
242 tied
243 female
244 straw
245 grilled
246 rolled
247 rounded
248 wrapped
249 attached
250 messy
251 bathroom
252 swimming
253 deep
254 pretty
255 sleeveless
256 granite
257 rainbow colored
258 fallen
259 modern
260 kneeling
261 kitchen
262 turned
263 used
264 murky
265 busy
266 snowboarding
267 still
268 soccer
269 mounted
270 antique
271 street
272 watching
273 slanted
274 faded
275 glowing
276 teal
277 gravel
278 balding
279 outdoor
280 heavy
281 cracked
282 snow
283 baked
284 fancy
285 transparent
286 roman
287 vintage
288 old fashioned
289 cooking
290 horizontal
291 crouched
292 shredded
293 computer
294 docked
295 navy
296 octagonal
297 shallow
298 sparse
299 hard
300 cotton
301 fat
302 waving
303 carpeted
304 electronic
305 protective
306 foggy
307 rippled
308 cloth
309 squatting
310 shaggy
311 scattered
312 new
313 barren
314 stained
315 wireless
316 designed
317 reading
318 overgrown
319 apple
320 looking down
321 sliding
322 ocean
323 plush
324 tilted
325 shaded
326 dusty
327 flowered
328 crumpled
329 dotted
330 mesh
331 collared
332 traffic
333 woven
334 athletic
335 lighted
336 power
337 clay
338 made
339 crispy
340 elderly
341 bunched
342 warm
343 cluttered
344 brunette
345 textured
346 spread
347 brass
348 chinese
349 styrofoam
350 elevated
351 palm
352 blowing
353 copper
354 foamy
355 unripe
356 cheese
357 sheer
358 padded
359 half full
360 shaped
361 laughing
362 winter
363 manicured
364 indoors
365 speckled
366 batting
367 loose
368 caucasian
369 peeling
370 diced
371 dense
372 adidas
373 military
374 tin
375 halved
376 jeans
377 wired
378 mixed
379 bronze
380 alert
381 neat
382 public
383 drawn
384 cobblestone
385 massive
386 real
387 peeled
388 assorted
389 sunlit
390 chipped
391 blooming
392 american
393 perched
394 shaved
395 spiky
396 wool
397 bamboo
398 gloved
399 spraying
400 commercial
401 steep
402 braided
403 steamed
404 tight
405 partly cloudy
406 unpeeled
407 carved
408 damaged
409 edged
410 directional
411 strong
412 pulled back
413 wrinkly
414 muscular
415 tabby
416 mowed
417 disposable
418 portable
419 shut
420 looking up
421 sad
422 packed
423 rippling
424 vanilla
425 eaten
426 barbed
427 safety
428 shingled
429 roasted
430 juicy
431 pine
432 shadowed
433 diamond
434 crystal
435 angled
436 weathered
437 homemade
438 seasoned
439 paneled
440 sloped
441 sweet
442 miniature
443 domed
444 pizza
445 torn
446 crowded
447 tail
448 beaded
449 suspended
450 cushioned
451 tangled
452 stormy
453 blank
454 tasty
455 hitting
456 deciduous
457 frozen
458 license
459 fenced
460 sprinkled
461 feathered
462 dull
463 tropical
464 twisted
465 crooked
466 wine
467 browned
468 cylindrical
469 greasy
470 abandoned
471 capital
472 toilet
473 unlit
474 iced
475 upholstered
476 gloomy
477 professional
478 creamy
479 analog
480 jagged
481 oblong
482 ruffled
483 connected
484 ivory
485 christmas
486 upper
487 coffee
488 lace
489 ridged
490 comfortable
491 foreign
492 knotted
493 french
494 support
495 performing trick
496 artificial
497 spiral
498 crumbled
499 dangling
500 cordless
501 wedding
502 fire
503 rustic
504 trash
505 uneven
506 grated
507 curvy
508 telephone
509 unoccupied
510 curious
511 rotten
512 oriental
513 folding
514 crusty
515 rimmed
516 birthday
517 polished
518 funny
519 crashing
520 powdered
521 lower
522 winding
523 calico
524 banana
525 angry
526 inflated
527 scrambled
528 wii
529 vast
530 sleepy
531 rainy
532 groomed
533 glossy
534 wooded
535 sturdy
536 vibrant
537 misty
538 beer
539 office
540 melting
541 silk
542 discolored
543 corded
544 formal
545 vacant
546 oversized
547 vinyl
548 spinning
549 simple
550 translucent
551 mature
552 tomato
553 crisp
554 boiled
555 recessed
556 gas
557 handmade
558 clumped
559 railroad
560 rugged
561 sculpted
562 burning
563 patched
564 industrial
565 chubby
566 ugly
567 garbage
568 powerful
569 elongated
570 pepper
571 abundant
572 opaque
573 ornamental
574 fluorescent
575 soda
576 packaged
577 inflatable
578 unpaved
579 soap
580 strawberry
581 park
582 exterior
583 intricate
584 sealed
585 quilted
586 hollow
587 breakable
588 pastel
589 urban
590 wrist
591 breaking
592 regular
593 forested
594 abstract
595 unhappy
596 incomplete
597 fine
598 chalk
599 coarse
600 polo
601 unhealthy
602 irregular
603 polar
604 uncomfortable
605 typical
606 complete
607 scarce
608 immature
"""

""" RELATIONS
0 right
1 left
2 on
3 wearing
4 near
5 in
6 of
7 behind
8 in front of
9 under
10 holding
11 with
12 by
13 on the side of
14 watching
15 inside
16 carrying
17 eating
18 at
19 riding
20 playing
21 covered by
22 covering
23 touching
24 on the front of
25 full of
26 riding on
27 using
28 covered in
29 sitting at
30 playing with
31 crossing
32 on the back of
33 surrounded by
34 reflected in
35 covered with
36 flying
37 pulled by
38 contain
39 standing by
40 driving on
41 surrounding
42 walking down
43 printed on
44 attached to
45 talking on
46 facing
47 driving
48 worn on
49 floating in
50 grazing on
51 on the bottom of
52 standing in front of
53 topped with
54 playing in
55 walking with
56 swimming in
57 driving down
58 pushed by
59 playing on
60 close to
61 waiting for
62 between
63 running on
64 tied to
65 on the edge of
66 holding onto
67 talking to
68 eating from
69 perched on
70 parked by
71 reaching for
72 connected to
73 grazing in
74 floating on
75 wrapped around
76 skiing on
77 stacked on
78 parked at
79 standing at
80 walking across
81 plugged into
82 stuck on
83 stuck in
84 drinking from
85 seen through
86 sitting by
87 sitting in front of
88 looking out
89 parked in front of
90 petting
91 wrapped in
92 selling
93 parked along
94 coming from
95 lying inside
96 sitting inside
97 walking by
98 sitting with
99 making
100 walking through
101 hung on
102 walking along
103 going down
104 leaving
105 running in
106 flying through
107 mounted to
108 on the other side of
109 licking
110 sniffing
111 followed by
112 following
113 riding in
114 biting
115 parked alongside
116 chasing
117 leading
118 hanging off
119 helping
120 coming out of
121 hanging out of
122 staring at
123 walking toward
124 served on
125 hugging
126 entering
127 skiing in
128 looking in
129 draped over
130 tied around
131 exiting
132 looking down at
133 looking into
134 drawn on
135 balancing on
136 jumping over
137 posing with
138 reflecting in
139 eating at
140 walking up
141 sewn on
142 reflected on
143 about to hit
144 getting on
145 observing
146 approaching
147 traveling on
148 walking towards
149 wading in
150 growing by
151 displayed on
152 growing along
153 mixed with
154 grabbing
155 jumping on
156 scattered on
157 climbing
158 pointing at
159 opening
160 taller than
161 going into
162 decorated by
163 decorating
164 preparing
165 coming down
166 tossing
167 eating in
168 growing from
169 chewing
170 washing
171 herding
172 skiing down
173 looking through
174 picking up
175 trying to catch
176 standing against
177 looking over
178 typing on
179 piled on
180 tying
181 shining through
182 smoking
183 cleaning
184 walking to
185 smiling at
186 guiding
187 dragging
188 chained to
189 going through
190 enclosing
191 adjusting
192 running through
193 skating on
194 photographing
195 smelling
196 kissing
197 falling off
198 decorated with
199 walking into
200 worn around
201 walking past
202 blowing out
203 towing
204 jumping off
205 sprinkled on
206 moving
207 running across
208 hidden by
209 traveling down
210 splashing
211 hang from
212 looking toward
213 kept in
214 cooked in
215 displayed in
216 sitting atop
217 brushing
218 buying
219 in between
220 larger than
221 smaller than
222 pouring
223 playing at
224 longer than
225 higher than
226 jumping in
227 shorter than
228 bigger than
"""