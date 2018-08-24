/**
 * Incredibly crude way of allowing the user to hide type signatures.
 */
$(function(){  
    var markTypeSignatures = function($code){
        var encounteredTypes = false;        

        // 
        var $block = $code.children().first();
        
        // 
        if (!$block.hasClass("nf")){
            $block = $block.nextAll(".nf").first();
        }
        
        // 
        while ($block.length){
            var $blockElements = $block.nextUntil(".nf");
                       
            // Gather all nodes (including text nodes!) between this function
            // definition and the next.
            var node = $block.get(0);
            var last = $blockElements.nextAll(".nf").get(0);
            var blockContents = [];
            
            while (node !== null && node !== last){
                blockContents.push(node);
                node = node.nextSibling;
            }
            
            // Check if this is a type signature. If so, wrap it in <type>
            if ($blockElements.filter(":contains('::')").length){
                var $type = $("<type>");
                $blockElements.first().before($type);
                $.map(blockContents, function(e){
                    $type.append(e);
                });
                
                encounteredTypes = true;
            }
            
            // Continue to next block (if any)
            $block = $(last);
        }
        
        if (encounteredTypes){
            $code.prepend($("<span class='hide-signature'></span>").click(function(){
               $(this).toggleClass("hide-signature").toggleClass("show-signature");
               $("type", $code).toggleClass("hidden");
            }));
        }
    }
    
    // Mark all type signatures with <span class="type">...</span> in Haskell code elements
    $("code.language-haskell").each(function(_, code){
        markTypeSignatures($(code));
    });
});




