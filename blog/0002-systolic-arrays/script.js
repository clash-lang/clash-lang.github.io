var SystolicArray = function(selector, positions, drawCell, cleanPEs){
    var self = {};

    self.drawCell = drawCell;
    self.cleanPEs = false;

    self.$sysarray = $(selector);
    self.$matrix = self.$sysarray.find("table");
    self.$next = self.$sysarray.find(".next");
    self.$reset = self.$sysarray.find(".reset");

    self.maxRight = 7;
    self.maxBottom = 7;

    self.dimensions = {
        m: 8,
        n: 13
    }
    
    self.sysLocation = {
        x: 5,
        y: 5
    }

    self.initPositions = positions;
    self.positions = $.extend(true, {}, self.initPositions);

    self.init = function(){
        // Create empty table
        var emptyColumns = "<td></td>".repeat(self.dimensions.n);
        var emptyRows = ("<tr>" + emptyColumns + "</tr>").repeat(self.dimensions.m);
        self.$matrix.html(emptyRows);

        // Set systolic array markers
        var x = self.sysLocation.x;
        var y = self.sysLocation.y;
        
        for (var i=x; i < x+3; i++){
            for (var j=y; j < y+3; j++){
                self.$getCell(i, j).addClass("pe"); 
            }
        }

        // Draw elements for the first time
        self.draw(true);

        self.$next.click(function(){
            self.step();
            self.draw(self.cleanPEs);
        });

        self.$reset.click(function(){
            self.reset();
            self.draw(true);
        });
    }

    self.getCell = function(matrix, i, j){
        if (matrix[i] === undefined || matrix[i][j] === undefined){
            return null;
        } else {
            return matrix[i][j];
        }
    }

    self.$getCell = function(i, j){
        return self.$matrix.find("tr:nth-child(" + (i+1) + ") td:nth-child(" + (j+1) + ")");
    }

    self.$setCell = function(i, j, value){
        self.$getCell(i, j).html(value);
    }

    self._draw = function(i, j){
        var a = self.getCell(self.positions.lefts, i, j);
        var b = self.getCell(self.positions.tops, i, j);

        if (a !== null && b !== null){
            var prevValue = self.$getCell(i, j).html();
            self.$setCell(i, j, self.drawCell.bind(self)(a, b, i, j, prevValue));
        } else if (a !== null){
            self.$getCell(i, j).addClass("a");
            self.$setCell(i, j, a);
        } else if (b !== null){
            self.$getCell(i, j).addClass("b");
            self.$setCell(i, j, b);
        } else {
            alert("Unreachable code in self._draw() reached");
        }
    }

    self.draw = function(clean){
        var tds = self.$matrix.find("td");

        if (!clean){
            tds = tds.not(".pe");
        }

        tds.html("").removeClass("a b");
        self.mapPositions(self._draw);
    }

    self.mapPositions = function(f){
        var lefts = self.positions.lefts;
        var tops = self.positions.tops;

        self.mapCells(lefts, function(i, j){
            f(parseInt(i), parseInt(j));
        });

        self.mapCells(tops, function(i, j){
            if (lefts[i] === undefined || lefts[i][j] === undefined){
                f(parseInt(i), parseInt(j));
            }
        });
    }

    self.mapCells = function(matrix, f){
        for (var i in matrix){
            var row = matrix[i];
            for (var j in row){
                var value = row[j];
                f(parseInt(i), parseInt(j), value);
            }
        }
    }

    self.step = function(){
        // Move left values right
        var newLefts = {};
        self.mapCells(self.positions.lefts, function(i, j, value){
            if (j >= self.maxRight) return;
            if (!newLefts[i]) newLefts[i] = {};
            newLefts[i][j+1] = value;                 
        });

        // Move top values down 
        var newTops = {};
        self.mapCells(self.positions.tops, function(i, j, value){
            if (i >= self.maxBottom) return;
            if (!newTops[i+1]) newTops[i+1] = {};
            newTops[i+1][j] = value;                 
        });

        self.positions = {
            lefts: newLefts,
            tops: newTops
        }
    }

    self.reset = function(){
        self.positions = $.extend(true, {}, self.initPositions);
    }

    return self;

};

// MM0
$(function() {
    var mm0 = SystolicArray("#mm0", {
        lefts: {
            3: {0: "0 →", 1: "1 →", 2: "2 →"},
            4: {0: "3 →", 1: "4 →", 2: "5 →"},
            5: {0: "6 →", 1: "7 →", 2: "8 →"}
        },
        tops: {
            0: {3: "A ↓", 4: "B ↓", 5: "C ↓"},
            1: {3: "D ↓", 4: "E ↓", 5: "F ↓"},
            2: {3: "G ↓", 4: "H ↓", 5: "I ↓"}
        }
    }, function(a, b){
        return "(<span class='b'>" + b + "</span>, <span class='a'>" + a + "</span>)";
    });
    
    mm0.cleanPEs = true;
    mm0.sysLocation = {x: 3, y: 3};
    mm0.dimensions = {m: 6, n: 9};
    mm0.maxRight = 5;
    mm0.init();
});

// MM1
$(function() {
    var mm1 = SystolicArray("#mm1", {
        lefts: {
            5: {2: "0 →", 3: "1 →", 4: "2 →"},
            6: {1: "3 →", 2: "4 →", 3: "5 →"},
            7: {0: "6 →", 1: "7 →", 2: "8 →"}
        },
        tops: {
            0: {7: "C ↓"},
            1: {7: "F ↓", 6: "B ↓"},
            2: {7: "I ↓", 6: "E ↓", 5: "A ↓"},
            3: {6: "H ↓", 5: "D ↓"},
            4: {5: "G ↓"}
        }
    }, function(a, b){
        return "(<span class='a'>" + a + "</span>, <span class='b'>" + b + "</span>)";
    });
    
    mm1.cleanPEs = true;
    mm1.init();
});

// MM1
$(function() {
    var mm1 = SystolicArray("#mm1-rep", {
        lefts: {
            5: {2: "0", 3: "1", 4: "2"},
            6: {1: "3", 2: "4", 3: "5"},
            7: {0: "6", 1: "7", 2: "8"}
        },
        tops: {
            0: {7: "C"},
            1: {7: "F", 6: "B"},
            2: {7: "I", 6: "E", 5: "A"},
            3: {6: "H", 5: "D"},
            4: {5: "G"}
        }
    }, function(a, b){
        return "(<span class='a'>" + a + "</span>, <span class='b'>" + b + "</span>)";
    });
    
    mm1.cleanPEs = true;
    mm1.init();
});

// MM2 
$(function() {
    var mm2 = SystolicArray("#mm2", {
        lefts: {
            5: {2: "a02", 3: "a01", 4: "a00"},
            6: {1: "a12", 2: "a11", 3: "a10"},
            7: {0: "a22", 1: "a21", 2: "a20"}
        },
        tops: {
            0: {7: "b22"},
            1: {7: "b12", 6: "b21"},
            2: {7: "b02", 6: "b11", 5: "b20"},
            3: {6: "b01", 5: "b10"},
            4: {5: "b00"}
        }
    }, function(a, b, i, j, prevValue){
        var value = "";

        if (prevValue){
            value = prevValue + " +<br/>";
        }

        return value + "<span class='a'>" + a + "</span>" + "*" + "<span class='b'>" + b + "</span>";
    });

    mm2.init();
});


// MM3 
$(function() {
    var mm3 = SystolicArray("#mm3", {
        lefts: {
            5: {2: "1", 3: "2", 4: "1"},
            6: {1: "0", 2: "1", 3: "0"},
            7: {0: "4", 1: "3", 2: "2"}
        },
        tops: {
            0: {7: "1"},
            1: {7: "1", 6: "8"},
            2: {7: "1", 6: "7", 5: "1"},
            3: {6: "5", 5: "6"},
            4: {5: "2"}
        }
    }, function(a, b, i, j, prevValue){
        var value = "";

        if (prevValue){
            value = prevValue + " +<br/>";
        }

        return value + "<span class='a'>" + a + "</span>" + "*" + "<span class='b'>" + b + "</span>";
    });

    mm3.init();
});



// MM4
$(function() {
    var mm4 = SystolicArray("#mm4", {
        lefts: {
            5: {2: "1", 3: "2", 4: "1"},
            6: {1: "0", 2: "1", 3: "0"},
            7: {0: "4", 1: "3", 2: "2"}
        },
        tops: {
            0: {7: "1"},
            1: {7: "1", 6: "8"},
            2: {7: "1", 6: "7", 5: "1"},
            3: {6: "5", 5: "6"},
            4: {5: "2"}
        }
    }, function(a, b, i, j, prevValue){
        var value;
        value  = parseInt(prevValue) || 0;
        value += a * b;
        return value;
    });

    mm4.init();
});
