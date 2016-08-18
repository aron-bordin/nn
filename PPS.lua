local PPS, parent = torch.class('nn.PPS', 'nn.Module')


function PPS:__init(order)
    parent.__init(self)
    self.order = order or 0
    assert(self.order <= 4, 'This order is not implemented. Use <= 4')
end

function PPS:updateOutput(input)
    input.THNN.PPS_updateOutput(
        input:cdata(),
        self.output:cdata(),
        self.order
    )
    return self.output
end

function PPS:updateGradInput(input, gradOutput)
    input.THNN.PPS_updateGradInput(
        input:cdata(),
        gradOutput:cdata(),
        self.gradInput:cdata(),
        self.output:cdata(),
        self.order + 1
    )
    return self.gradInput
end
