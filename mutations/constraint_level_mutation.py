from abc import abstractmethod
from typing import Callable

from mutations.mutation import Mutation


class ConstrainLevelMutation(Mutation):

    def __init__(self, column,name):
        self.column = column
        self.name=name

    def apply(self, generator):
        return self.mutate_constrains(generator)

    @abstractmethod
    def mutate_constrains(self):
        pass


class ReplaceColumnConstraintMutation(ConstrainLevelMutation):
    def __init__(self, column: str,  new_constraint: Callable,name:str):
        super().__init__(column,name)
        self.constraints = new_constraint

    def mutate_constrains(self, generator):
        def is_related_to_column(constraint):
            return self.column in getattr(constraint, "__name__", "") \
                   or self.column in getattr(constraint, "__doc__", "")

        new_constraints = [
            c for c in generator.constraints
            if not is_related_to_column(c)
        ]

        new_constraints.append(self.constraints)

        return generator.__class__(
            data=generator.data,
            seed=generator.seed,
            epochs=generator.epochs,
            constraints=new_constraints
        )

class RemoveColumnConstraintMutation(ConstrainLevelMutation):

    def __init__(self, column: str,  on: int,name:str):
        super().__init__(column,name)
        self.on = on

    def mutate_constrains(self, generator):
        def is_related_to_column(constraint):
            if self.on==0: return False
            return self.column in getattr(constraint, "__name__", "") \
                   or self.column in getattr(constraint, "__doc__", "")

        new_constraints = [
            c for c in generator.constraints
            if not is_related_to_column(c)
        ]

        return generator.__class__(
            data=generator.data,
            seed=generator.seed,
            epochs=generator.epochs,
            constraints=new_constraints
        )

